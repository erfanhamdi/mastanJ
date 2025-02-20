import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node

F = Frame()
F.set_cross_section(E=200, Iy=1, Iz=1, J=1, nu=0.3)
node1 = Node(coords = np.array([0, 0, 0])) # node a
node2 = Node(coords = np.array([4, -4, 0])) # node b
node3 = Node(coords = np.array([-6.928, -4, 0])) # node c
elem1 = Element(node_list=[node1, node2], A = 20e3)
elem2 = Element(node_list=[node2, node3], A = 18e3)
elem3 = Element(node_list=[node1, node3], A = 15e3)

def test_adj_matrix():
    adj_mat = F.create_adjacency_matrix([elem1, elem2, elem3])
    assert np.all(adj_mat == np.array([[1., 1., 0., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]]))
    print("Passed test_adj_matrix")

def test_local_stiffness_matrix():
    K = elem1.global_stiffness_mat()
    print(K)