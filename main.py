import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node
from src.shape_functions import plot_original, plot_deformed, LinearShapeFunctions, HermiteShapeFunctions

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
    # element0 = Element(node_list=[node0, node4], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element1 = Element(node_list=[node1, node5], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element2 = Element(node_list=[node2, node6], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element3 = Element(node_list=[node3, node7], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element4 = Element(node_list=[node4, node5], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element5 = Element(node_list=[node5, node6], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element6 = Element(node_list=[node6, node7], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
    # element7 = Element(node_list=[node7, node4], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho, local_z = local_z)
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
    # print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    # print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")

    L = 50
    E = 1000
    nu = 0.3
    r = 1
    A = np.pi*r**2
    I_y = np.pi*r**4 / 4
    I_z = np.pi*r**4 / 4
    I_rho = np.pi*r**4 / 2
    J = np.pi*r**4 / 2
    P_analytical = np.pi **2 * E * I_z / (2 * L) **2
    F6 = Frame()
    node0 = Node(coords = np.array([0, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    node1 = Node(coords = np.array([30, 40, 0]), F_x = -3/5, F_y = -4/5, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    element0 = Element(node_list=[node0, node1], A = A, E = E, Iz = I_z, Iy= I_y, J = J, nu = nu, I_rho = I_rho)
    F6.add_elements([element0])
    F6.assemble()
    delta, F_rxn = F6.solve()
    print(f"Delta:\n {delta}")
    print(f"Reaction force on supports:\n {F_rxn}")
    hermite_sf = HermiteShapeFunctions()
    # plot_deformed([element0], hermite_sf, delta, scale=10)
    F6.assemble_geometric()
    eigvals, eigvecs = F6.eigenvalue_analysis()
    critical_load_idx = np.argsort(abs(eigvals.real))[0]
    print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")
    import matplotlib.pyplot as plt
    plt.plot(eigvecs[:, critical_load_idx])
    plt.show()