import numpy as np

class Frame:
    # This class variable will be shared by all subclasses.
    E = None
    A = None
    L = None
    Iy = None
    Iz = None
    J = None
    nu = None

    @classmethod
    def set_cross_section(cls, E, A, Iy, Iz, J, nu):
        cls.E = E
        cls.A = A
        cls.Iy = Iy
        cls.Iz = Iz
        cls.J = J
        cls.nu = nu

    def add_elements(self, elems):
        self.elems = elems
        self.nodes = []
        for elem in elems:
            for node in elem.node_list:
                if node not in self.nodes:
                    self.nodes.append(node)
        self.number_of_dofs = self.num_dofs()

    def are_elems_connected(self, elem_1, elem_2):
        # check if two elems have a common node object
        for node in elem_1.node_list:
            if node in elem_2.node_list:
                return 1
        return 0
    
    def adjacency_matrix(self, elems):
        adj_mat = np.zeros((len(elems)+1, len(elems)+1))
        for elem in elems:
            node_1_id = elem.node_list[0].id
            node_2_id = elem.node_list[1].id
            adj_mat[node_1_id, node_2_id] = 1
            adj_mat[node_2_id, node_1_id] = 1
            adj_mat[node_1_id, node_1_id] = 1
            adj_mat[node_2_id, node_2_id] = 1
        return adj_mat

    def adjacency_filter(self, adj_mat):
        adj_filter = np.zeros((6 * adj_mat.shape[0], 6 * adj_mat.shape[1]))
        for i in range(adj_mat.shape[0]):
            for j in range(adj_mat.shape[1]):
                adj_filter[6*i:6*(i+1), 6*j:6*(j+1)] = adj_mat[i, j]
        return adj_filter
    
    def num_dofs(self):
        node_dict = {}
        for node in self.nodes:
            node_dict[node.id] = 1
        return 6 * len(node_dict)
    
    def assemble(self):
        self.K = np.zeros((self.number_of_dofs, self.number_of_dofs))
        self.F = np.zeros((self.number_of_dofs, 1))
        self.Delta = np.zeros((self.number_of_dofs, 1))
        for i, elem in enumerate(self.elems):
            small_K = elem.global_stiffness_mat()
            node_1_id = elem.node_list[0].id
            node_2_id = elem.node_list[1].id
            self.K[6*node_1_id:6*(node_1_id+1), 6*node_1_id:6*(node_1_id+1)] += small_K[:6, :6]
            self.K[6*node_1_id:6*(node_1_id+1), 6*node_2_id:6*(node_2_id+1)] += small_K[:6, 6:]
            self.K[6*node_2_id:6*(node_2_id+1), 6*node_1_id:6*(node_1_id+1)] += small_K[6:, :6]
            self.K[6*node_2_id:6*(node_2_id+1), 6*node_2_id:6*(node_2_id+1)] += small_K[6:, 6:]

    def partition(self,):
        free_dofs = []
        fixed_dofs = []
        f_list = []
        delta_list = []
        for node in self.nodes:
            free_dofs += [i + 6*node.id for i in node.free_dofs]
            fixed_dofs += [i + 6*node.id for i in node.fixed_dofs]
            f_list += node.load
            delta_list += node.bc
        self.K_ff = self.K[np.ix_(free_dofs, free_dofs)]
        self.K_fs = self.K[np.ix_(free_dofs, fixed_dofs)]
        self.K_sf = self.K[np.ix_(fixed_dofs, free_dofs)]
        self.K_ss = self.K[np.ix_(fixed_dofs, fixed_dofs)]
        self.F_f = np.array(f_list)[free_dofs]
        # self.F_s = np.array(f_list)[fixed_dofs]
        # self.Delta_f = np.array(delta_list)[free_dofs]
        # Delta_s = np.array(delta_list)[fixed_dofs]

    def solve(self,):
        self.Delta_f = np.linalg.pinv(self.K_ff) @ self.F_f
        self.F_s = self.K_sf @ self.Delta_f
        return self.Delta_f, self.F_s
    

        