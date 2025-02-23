import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node

F = Frame()
node1 = Node(coords = np.array([0, 0, 0])) # node a
node2 = Node(coords = np.array([4, -4, 0])) # node b
node3 = Node(coords = np.array([-6.928, -4, 0])) # node c
elem1 = Element(node_list=[node1, node2], A = 20e3, E = 200, Iy = 1, Iz = 1, J = 1, nu = 0.3)
elem2 = Element(node_list=[node2, node3], A = 18e3, nu = 0.3, E = 200, Iy = 1, Iz = 1, J = 1)
elem3 = Element(node_list=[node1, node3])

def test_element_params():
    assert elem1.E == 200
    assert elem1.A == 20e3
    assert elem1.nu == 0.3
    assert elem1.Iy == 1
    assert elem1.Iz == 1
    assert elem1.J == 1

def test_element_length():
    assert elem1.L == np.sqrt(2*4**2)

