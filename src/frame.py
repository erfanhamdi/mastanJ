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
    def num_dofs(self, elems):
        node_dict = {}
        for elem in elems:
            for node in elem.node_list:
                node_dict[node.id] = 1
        return 6 * len(node_dict)
    def assemble(self, elems):
        n_dofs = self.num_dofs(elems)
        K = np.zeros((n_dofs, n_dofs))
        for i, elem in enumerate(elems):
            small_K = elem.global_stiffness_mat()
            node_1_id = elem.node_list[0].id
            node_2_id = elem.node_list[1].id
            K[6*node_1_id:6*(node_1_id+1), 6*node_1_id:6*(node_1_id+1)] += small_K[:6, :6]
            K[6*node_1_id:6*(node_1_id+1), 6*node_2_id:6*(node_2_id+1)] += small_K[:6, 6:]
            K[6*node_2_id:6*(node_2_id+1), 6*node_1_id:6*(node_1_id+1)] += small_K[6:, :6]
            K[6*node_2_id:6*(node_2_id+1), 6*node_2_id:6*(node_2_id+1)] += small_K[6:, 6:]
        return K
