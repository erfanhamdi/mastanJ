import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node

F = Frame()
F.set_cross_section(E=1, A=1, L=1, Iy=1, Iz=1, J=1, nu=1)
node1 = Node(coords = np.array([0, 0, 0]))
node2 = Node(coords = np.array([0, 1, 0]))
elem1 = Element(node_list=[node1, node2])

