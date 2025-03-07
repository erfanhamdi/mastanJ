import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node
from src.shape_functions import plot_mode_shape, HermiteShapeFunctions

if __name__ == "__main__":
##################################### Problem 1 #################################################
    r = 1
    E = 10000  
    A = np.pi*r**2
    I_y = np.pi*r**4 / 4
    I_z = np.pi*r**4 / 4
    I_rho = np.pi*r**4 / 2
    J = np.pi*r**4 / 2
    nu = 0.3
    x1, y1, z1 = 18, 56, 44
    x0, y0, z0 = 0, 0, 0
    F8 = Frame()
    node0 = Node(coords = np.array([0, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    node6 = Node(coords = np.array([x1, y1, z1]), F_x = 0.05, F_y = -0.1, F_z = 0.23, M_x = 0.1, M_y = -0.025, M_z = -0.08)
    x_ = np.linspace(0, x1, 7)[1:-1]
    y_ = np.linspace(0, y1, 7)[1:-1]
    z_ = np.linspace(0, z1, 7)[1:-1]
    print(x_, y_, z_)
    interm_nodes = [node0]
    for i in range(5):
        interm_nodes.append(Node(coords = np.array([x_[i], y_[i], z_[i]]), F_x = 0, F_y = 0, F_z = 0, M_x = 0, M_y = 0, M_z = 0))
    interm_nodes.append(node6)
    elems = []
    for i in range(6):
        elems.append(Element(node_list=[interm_nodes[i], interm_nodes[i+1]], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho))
    F8.add_elements(elems)
    F8.assemble()
    delta, F_rxn = F8.solve()
    for i in range(len(interm_nodes)):
        print(f"delta node {i}: {delta[i * 6 : (i + 1) * 6]}")
        print(f"Force/Momments on node {i}:\n {F_rxn[i * 6 : (i + 1) * 6]}")

##################################### Problem 2 #################################################
    P = 1
    L = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
    Fx_2 = -1 * P * (x1 - x0) / L
    Fy_2 = -1 * P * (y1 - y0) / L
    Fz_2 = -1 * P * (z1 - z0) / L
    F9 = Frame()
    node0 = Node(coords = np.array([x0, y0, z0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    node6_ = Node(coords = np.array([x1, y1, z1]), F_x = Fx_2, F_y = Fy_2, F_z = Fz_2, M_x = 0, M_y = 0, M_z = 0)
    interm_nodes[-1] = node6_
    elems = []
    for i in range(6):
        elems.append(Element(node_list=[interm_nodes[i], interm_nodes[i+1]], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho))
    F9.add_elements(elems)
    F9.assemble()
    delta, F_rxn = F9.solve()
    hermite_sf = HermiteShapeFunctions()
    F9.assemble_geometric()
    eigvals, eigvecs = F9.eigenvalue_analysis()
    critical_load_idx = np.argsort(abs(eigvals.real))[0]
    mode_shape = eigvecs[:, critical_load_idx]
    eigvec_array = np.zeros(len(F9.free_dofs) + len(F9.fixed_dofs))
    eigvec_array[F9.free_dofs] = mode_shape
    eigvec_array[F9.fixed_dofs] = 0
    print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")

##################################### Problem 3 #################################################
    L1 = 15.0
    L2 = 30.0
    L3 = 14.0
    L4 = 16.0
    r = 1
    b = 0.5
    h = 1
    elem_type_1 = {"E": 10000, "nu": 0.3, "A": np.pi * r**2, "Iz": np.pi * r**4/4, "Iy": np.pi * r**4/4, "J": np.pi * r**4 /2, "I_rho": np.pi * r**4 /2}
    elem_type_2 = {"E": 50000, "nu": 0.3, "A": b * h, "Iz": b * h **3 /12, "Iy": h * b**3 / 12, "J": 0.028610026041666667, "I_rho": b * h / 12 * (b**2 + h**2)}
    F10 = Frame()
    z_ = [0 , L3, L3 + L4]
    y_ = [0, L2]
    x_ = [0, L1]
    nodes = []
    node_id = 0
    for zz in z_:
        print(f"@ z = {zz}")
        for yy in y_:
            for xx in x_:
                if zz == 0:
                    bc = {"u_x": 0, "u_y": 0, "u_z": 0, "theta_x": 0, "theta_y": 0, "theta_z": 0}
                if zz == L3:
                    bc = {"F_x": 0, "F_y": 0, "F_z": 0, "M_x": 0, "M_y": 0, "M_z": 0}
                if zz == L3 + L4:
                    bc = {"F_x": 0, "F_y": 0, "F_z": -1, "M_x": 0, "M_y": 0, "M_z": 0}
                nodes.append(Node(coords = np.array([xx, yy, zz]), **bc))
                print(f"added node {node_id}", xx, yy, zz, "with bc", bc)
                node_id += 1
    connectivity_up = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11]]
    connectivity_side = [[4, 5], [6, 7], [8, 9], [10, 11], [4, 6], [5, 7], [8, 10], [9, 11]]
    elems = []
    for idx, conn in enumerate(connectivity_up):
        node_1 = nodes[conn[0]]
        node_2 = nodes[conn[1]]
        elem_type = elem_type_1
        elems.append(Element(node_list=[node_1, node_2], **elem_type))
    for idx, conn in enumerate(connectivity_side):
        node_1 = nodes[conn[0]]
        node_2 = nodes[conn[1]]
        elem_type = elem_type_2
        elems.append(Element(node_list=[node_1, node_2], **elem_type))
        print(f"added elem {idx}", node_1.coords, node_2.coords, "with elem_type", elem_type)
    print(f"added {len(elems)} elements")
    F10.add_elements(elems)
    F10.assemble()
    delta, F_rxn = F10.solve()
    print(f"Delta:\n {delta}")
    print(f"Reaction force on supports:\n {F_rxn}")
    hermite_sf = HermiteShapeFunctions()
    F10.assemble_geometric()
    eigvals, eigvecs = F10.eigenvalue_analysis()
    critical_load_idx = np.argsort(abs(eigvals.real))[0]
    mode_shape = eigvecs[:, critical_load_idx]
    eigvec_array = np.zeros(len(F10.free_dofs) + len(F10.fixed_dofs))
    eigvec_array[F10.free_dofs] = mode_shape
    eigvec_array[F10.fixed_dofs] = 0
    print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    # print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")
    plot_mode_shape(F10, F10.elems, hermite_sf, eigvecs[:, critical_load_idx], scale=10, discretization_points=20)