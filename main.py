import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node

if __name__ == "__main__":
    b = 0.5
    h = 1.0
    E = 1000
    nu = 0.3
    A = b*h
    Iy = h * b**3 / 12
    Iz = b * h**3 / 12
    I_polar = b*h/12 * (h**2 + b**2)
    J = 0.02861
    F3 = Frame()
    node0 = Node(coords = np.array([0, 0, 10]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0)
    node1 = Node(coords = np.array([15, 0, 10]), F_x = 0.1, F_y = 0.05, F_z = -0.07, M_x = 0.05, M_y = -0.1, M_z = 0.25)
    node2 = Node(coords = np.array([15, 0, 0]), u_x = 0, u_y = 0, u_z = 0, M_x = 0, M_y = 0, M_z = 0)
    element0 = Element(node_list=[node0, node1], A = A, E = E, Iz = Iz, Iy= Iy, J = J, nu = nu, local_z = np.array([0, 0, 1]))
    element1 = Element(node_list=[node1, node2], A = A, E = E, Iy = Iy, Iz = Iz, J = J, nu = nu, local_z = np.array([1, 0, 0]))
    F3.add_elements([element0, element1])
    F3.assemble()
    delta, F_rxn = F3.solve()
    print(f"Delta: {delta}")
    print(f"Reaction force on supports: {F_rxn}")
    