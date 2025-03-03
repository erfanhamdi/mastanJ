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
    b = 0.5
    h = 1.0
    E = 29000
    nu = 0.3
    A = 20
    Iy = 6.67
    Iz = 166.67
    I_polar = Iz + Iy
    J = 26.67
    F5 = Frame()
    node0 = Node(coords = np.array([0, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    node1 = Node(coords = np.array([20, 0, 0]), F_x = 0, F_y = 0, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    node2 = Node(coords = np.array([40, 0, 0]), F_x = 0, F_y = 0, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    node3 = Node(coords = np.array([60, 0, 0]), F_x = 0, F_y = -1, F_z = 0, M_x = 0, M_y = 0,  M_z = 0)
    element0 = Element(node_list=[node0, node1], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, I_rho = I_polar)
    element1 = Element(node_list=[node1, node2], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, I_rho = I_polar)
    element2 = Element(node_list=[node2, node3], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, I_rho = I_polar)
    F5.add_elements([element0, element1, element2])
    F5.assemble()
    delta, F_rxn = F5.solve()
    print(f"Delta:\n {delta}")
    print(f"Reaction force on supports:\n {F_rxn}")
    # hermite_sf = HermiteShapeFunctions()
    # plot_deformed([element0, element1], hermite_sf, delta, scale=10)
    F5.assemble_geometric()
    eigvals, eigvecs = F5.eigenvalue_analysis()
    critical_load_idx = np.argsort(abs(eigvals.real))[0]
    print(f"Critical buckling load:\n {abs(eigvals.real)[critical_load_idx]}")
    print(f"Buckling mode:\n {eigvecs[:, critical_load_idx]}")