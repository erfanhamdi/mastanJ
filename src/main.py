import numpy as np

from frame import Frame
from element import Element
from node import Node

if __name__ == "__main__":
    F = Frame()
    node1 = Node(coords = np.array([0, 0, 0]), F_x = 500*np.cos(40*np.pi/180), F_y = 500*np.sin(40*np.pi/180), F_z = 0, M_x = 0, M_y = 0, M_z = 0) # node a
    node2 = Node(coords = np.array([4, -4, 0]), F_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0) # node b
    node3 = Node(coords = np.array([-6.928, -4, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0) # node c
    elem1 = Element(node_list=[node1, node2], A = 20e3, E = 200, Iy = 0, Iz = 0, J = 0)
    elem2 = Element(node_list=[node2, node3], A = 18e3, E = 200, Iy = 0, Iz = 0, J = 0)
    elem3 = Element(node_list=[node1, node3], A = 15e3, E = 200, Iy = 0, Iz = 0, J = 0)
    F.add_elements([elem1, elem2, elem3])
    k_elem1 = elem1.global_stiffness_mat()
    k_elem2 = elem2.global_stiffness_mat()
    k_elem3 = elem3.global_stiffness_mat()
    F.assemble()
    F.partition()
    delta, F_rxn = F.solve()
    print(delta)
    print(F_rxn)
    # F1 = Frame()
    # F1.set_cross_section(E=1, A=1, L=1, Iy=1, Iz=1, J=1, nu=1)
    # node1 = Node(coords = np.array([0, 0, 0]), D = 0)
    # node2 = Node(coords = np.array([0, 0, 1]))
    # node3 = Node(coords = np.array([0, 0, 2]))
    # node4 = Node(coords = np.array([0, 1, 1]))
    # elem1 = Element(node_list=[node1, node2])
    # elem2 = Element(node_list=[node2, node3])
    # elem3 = Element(node_list=[node1, node4])
    # adj_mat = F1.adjacency_matrix([elem1, elem2])
    # K = F1.assemble([elem1, elem3])
    # print(adj_mat)
    # print(K)
    # print(elem1.stiffness_mat())
    # print(elem1.Gamma())