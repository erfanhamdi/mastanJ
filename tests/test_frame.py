import numpy as np
import pytest

from src.frame import Frame
from src.element import Element
from src.node import Node

F = Frame()
node1 = Node(coords = np.array([0, 0, 0]), F_x = 500*np.cos(40*np.pi/180), F_y = 500*np.sin(40*np.pi/180), u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0) # node a
node2 = Node(coords = np.array([4, -4, 0]), F_x = 0, theta_z = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0) # node b
node3 = Node(coords = np.array([-6.928, -4, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0) # node c
node4 = Node(coords = np.array([-6.928, 0, 0]), u_x = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0, M_z = 0) 
elem1 = Element(node_list=[node1, node2], A = 20e3, E = 200, Iy = 0, Iz = 0, J = 1, nu = 0.3)
elem2 = Element(node_list=[node2, node3], A = 18e3, E = 200, Iy = 0, Iz = 0, J = 1, nu = 0.3)
elem3 = Element(node_list=[node1, node3], A = 15e3, E = 200, Iy = 0, Iz = 0, J = 1, nu = 0.3)
elem4 = Element(node_list=[node1, node4], A = 20e3, E = 200, Iy = 1, Iz = 1, J = 1, nu = 0.3)
F.add_elements([elem1, elem2, elem3, elem4])
F.assemble()
delta, F_rxn = F.solve()

def test_element_length():
    # element length using two nodes (node1 and node2)
    expected_length = np.sqrt((4-0)**2 + (-4-0)**2 + (0-0)**2)
    assert np.isclose(elem1.L, expected_length)

def test_add_elements():
    # Create fresh nodes for the test.
    node_a = Node(coords=np.array([0, 0, 0]))
    node_b = Node(coords=np.array([3, 0, 0]))
    node_c = Node(coords=np.array([0, 4, 0]))
    
    # Create two elements with a common node (node_b)
    elem_a = Element(node_list=[node_a, node_b], A=100, E=210e3, Iy=2, Iz=2, J=1, nu=0.3)
    elem_b = Element(node_list=[node_b, node_c], A=150, E=210e3, Iy=2, Iz=2, J=1, nu=0.3)

    frame = Frame()
    frame.add_elements([elem_a, elem_b])
    
    # Check that all unique nodes are added.
    expected_node_ids = {node_a.id, node_b.id, node_c.id}
    actual_node_ids = {node.id for node in frame.nodes}
    assert actual_node_ids == expected_node_ids
    
    # Check that the number of degrees of freedom is correctly computed (6 DOFs per node).
    expected_dofs = 6 * len(expected_node_ids)
    assert frame.number_of_dofs == expected_dofs

def test_solve():
    expected_delta_dof = np.array([np.float64( -0.00043791)])
    expected_F_rxn = np.array([np.float64(3.83022222e+02)])
    assert np.allclose(delta[6], expected_delta_dof)
    assert np.allclose(F_rxn[0], expected_F_rxn)