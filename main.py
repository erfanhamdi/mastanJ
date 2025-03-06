import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node
from src.shape_functions import plot_mode_shape, HermiteShapeFunctions

if __name__ == "__main__":
    # r = 1
    # E = 500
    # nu = 0.3
    # A = np.pi * r**2
    # Iy = np.pi * r**4 / 4
    # Iz = np.pi * r**4 / 4
    # I_polar = np.pi * r**4 / 2
    # J = np.pi * r**4 / 2
    # F4 = Frame()
    # node0 = Node(coords = np.array([0, 0, 0]), F_x = 0, F_y = 0, u_z = 0, M_x = 0, M_y = 0, M_z = 0)
    # node1 = Node(coords = np.array([-5, 1, 10]), F_x = 0.1, F_y = -0.05, F_z = -0.075, M_x = 0, M_y = 0, M_z = 0)
    # node2 = Node(coords = np.array([-1, 5, 13]), F_x = 0, F_y = 0, F_z = 0, M_x = 0.5, M_y = -0.1, M_z = 0.3)
    # node3 = Node(coords = np.array([-3, 7, 11]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node4 = Node(coords = np.array([6, 9, 5]), u_x = 0, u_y = 0, u_z = 0, M_x = 0, M_y = 0, M_z = 0)
    # element0 = Element(node_list=[node0, node1], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu)
    # element1 = Element(node_list=[node1, node2], A = A, E = E, Iy = Iy, Iz = Iz, J = J, nu = nu)
    # element2 = Element(node_list=[node2, node3], A = A, E = E, Iy = Iy, Iz = Iz, J = J, nu = nu)
    # element3 = Element(node_list=[node2, node4], A = A, E = E, Iy = Iy, Iz = Iz, J = J, nu = nu)
    # F4.add_elements([element0, element1, element2, element3])
    # F4.assemble()
    # delta, F_rxn = F4.solve()
    # print(f"Delta:\n {delta}")
    # print(f"Reaction force on supports:\n {F_rxn}")
    # hermite_sf = HermiteShapeFunctions()
    # plot_deformed([element0, element1, element2, element3], hermite_sf, delta, scale=10)
    # b = 0.5
    # h = 1.0
    # E = 29000
    # nu = 0.3
    # A = 20
    # Iy = 6.67
    # Iz = 166.67
    # I_polar = Iz + Iy
    # J = 26.67
    # F5 = Frame()
    # node0 = Node(coords = np.array([0, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node1 = Node(coords = np.array([20, 0, 0]), F_x = 0, F_y = 0, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    # node2 = Node(coords = np.array([40, 0, 0]), F_x = 0, F_y = 0, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    # node3 = Node(coords = np.array([60, 0, 0]), F_x = 0, F_y = -1, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    # element0 = Element(node_list=[node0, node1], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, I_rho = I_polar)
    # element1 = Element(node_list=[node1, node2], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, I_rho = I_polar)
    # element2 = Element(node_list=[node2, node3], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, I_rho = I_polar)
    # F5.add_elements([element0, element1, element2])
    # F5.assemble()
    # delta, F_rxn = F5.solve()
    # print(f"Delta:\n {delta}")
    # print(f"Reaction force on supports:\n {F_rxn}")
    # hermite_sf = HermiteShapeFunctions()
    # plot_deformed([element0, element1, element2], hermite_sf, delta, scale=10)
    # F5.assemble_geometric()
    # eigvals, eigvecs = F5.eigenvalue_analysis()
    # critical_load_idx = np.argsort(abs(eigvals.real))[0]
    # print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    # print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")

    # E = 500
    # nu = 0.3
    # r = 0.5
    # A = np.pi*r**2
    # I_y = np.pi*r**4 / 4
    # I_z = np.pi*r**4 / 4
    # I_rho = np.pi*r**4 / 2
    # J = np.pi*r**4 / 2
    # local_z = None
    # F7 = Frame()
    # node0 = Node(coords = np.array([0, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node1 = Node(coords = np.array([10, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node2 = Node(coords = np.array([10, 20, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node3 = Node(coords = np.array([0, 20, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node4 = Node(coords = np.array([0, 0, 25]), F_x = 0, F_y = 0, F_z = -1, M_x = 0, M_y = 0, M_z = 0)
    # node5 = Node(coords = np.array([10, 0, 25]), F_x = 0, F_y = 0, F_z = -1, M_x = 0, M_y = 0, M_z = 0)
    # node6 = Node(coords = np.array([10, 20, 25]), F_x = 0, F_y = 0, F_z = -1, M_x = 0, M_y = 0, M_z = 0)
    # node7 = Node(coords = np.array([0, 20, 25]), F_x = 0, F_y = 0, F_z = -1, M_x = 0, M_y = 0, M_z = 0)
    # element0 = Element(node_list=[node0, node4], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element1 = Element(node_list=[node1, node5], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element2 = Element(node_list=[node2, node6], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element3 = Element(node_list=[node3, node7], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element4 = Element(node_list=[node4, node5], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element5 = Element(node_list=[node5, node6], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element6 = Element(node_list=[node6, node7], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # element7 = Element(node_list=[node7, node4], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # F7.add_elements([element0, element1, element2, element3, element4, element5, element6, element7])
    # F7.assemble()
    # delta, F_rxn = F7.solve()
    # print(f"Delta:\n {delta}")
    # print(f"Reaction force on supports:\n {F_rxn}")
    # hermite_sf = HermiteShapeFunctions()
    # # plot_deformed([element0, element1, element2, element3, element4, element5, element6, element7], hermite_sf, delta, scale=100)
    # F7.assemble_geometric()
    # eigvals, eigvecs = F7.eigenvalue_analysis()
    # print(eigvals)
    # critical_load_idx = np.argsort(abs(eigvals.real))[0]
    # mode_shape = eigvecs[:, critical_load_idx]
    # eigvec_array = np.zeros(len(F7.free_dofs) + len(F7.fixed_dofs))
    # eigvec_array[F7.free_dofs] = mode_shape
    # eigvec_array[F7.fixed_dofs] = 0
    # print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    # print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")
    # plot_mode_shape(F7, F7.elems, hermite_sf, -eigvecs[:, critical_load_idx], scale=5, discretization_points=20)


    # L = 50
    # E = 1000
    # nu = 0.3
    # r = 1
    # A = np.pi*r**2
    # I_y = np.pi*r**4 / 4
    # I_z = np.pi*r**4 / 4
    # I_rho = np.pi*r**4 / 2
    # J = np.pi*r**4 / 2
    # P_analytical = np.pi **2 * E * I_z / (2 * L) **2
    # F6 = Frame()
    # node0 = Node(coords = np.array([0, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    # node1 = Node(coords = np.array([30, 40, 0]), F_x = -3/5, F_y = -4/5, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    # element0 = Element(node_list=[node0, node1], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    # F6.add_elements([element0])
    # F6.assemble()
    # delta, F_rxn = F6.solve()
    # print(f"Delta:\n {delta}")
    # print(f"Reaction force on supports:\n {F_rxn}")
    # hermite_sf = HermiteShapeFunctions()
    # # plot_deformed([element0], hermite_sf, delta, scale=10)
    # F6.assemble_geometric()
    # eigvals, eigvecs = F6.eigenvalue_analysis()
    # critical_load_idx = np.argsort(abs(eigvals.real))[0]
    # mode_shape = eigvecs[:, critical_load_idx]
    # eigvec_array = np.zeros(len(F6.free_dofs) + len(F6.fixed_dofs))
    # eigvec_array[F6.free_dofs] = mode_shape
    # eigvec_array[F6.fixed_dofs] = 0
    # print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    # print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")
    # plot_mode_shape(F6, F6.elems, hermite_sf, eigvecs[:, critical_load_idx], scale=10, discretization_points=20)
    
    r = 1
    E = 10000
    nu = 0.3
    A = np.pi*r**2
    I_y = np.pi*r**4 / 4
    I_z = np.pi*r**4 / 4
    I_rho = np.pi*r**4 / 2
    J = np.pi*r**4 / 2
    x1, y1, z1 = 18, 56, 44
    x0, y0, z0 = 0, 0, 0
    F8 = Frame()
    node0 = Node(coords = np.array([x0, y0, z0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    node6 = Node(coords = np.array([x1, y1, z1]), F_x = 0.05, F_y = -0.1, F_z = 0.23, M_x = 0.1, M_y = -0.025, M_z = -0.08)
    x_ = np.linspace(0, x1, 7)[1:]
    y_ = np.linspace(0, y1, 7)[1:]
    z_ = np.linspace(0, z1, 7)[1:]
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
    print(delta)
    print(f"Reaction force on supports:\n {F_rxn}")
    print(f"delta node wise:\n{delta.reshape(-1, 6)}")
    print(f"Delta node 3: {delta[6 *6 : 6*6+ 6]}")

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
    print(f"Delta:\n {delta}")
    print(f"Reaction force on supports:\n {F_rxn}")
    hermite_sf = HermiteShapeFunctions()
    # plot_deformed([element0], hermite_sf, delta, scale=10)
    F9.assemble_geometric()
    eigvals, eigvecs = F9.eigenvalue_analysis()
    critical_load_idx = np.argsort(abs(eigvals.real))[0]
    mode_shape = eigvecs[:, critical_load_idx]
    eigvec_array = np.zeros(len(F9.free_dofs) + len(F9.fixed_dofs))
    eigvec_array[F9.free_dofs] = mode_shape
    eigvec_array[F9.fixed_dofs] = 0
    print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")

    L1 = 15.0
    L2 = 30.0
    L3 = 14.0
    L4 = 16.0
    r = 1
    b = 0.5
    h = 1
    elem_type_1 = {"E": 10000, "nu": 0.3, "A": np.pi * r**2, "Iz": np.pi * r**4/4, "Iy": np.pi * r**4/4, "J": np.pi * r**4 /2, "I_rho": np.pi * r**4 /2}
    elem_type_2 = {"E": 50000, "nu": 0.3, "A": b * h, "Iz": b * h **3 /12, "Iy": h * b**3 / 12, "J": 0.028610026041666667, "I_rho": b * h / 12 * (b**2 + h**2), "local_z": [0,0,1]}
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
    print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")
    plot_mode_shape(F10, F10.elems, hermite_sf, eigvecs[:, critical_load_idx], scale=10, discretization_points=20)