import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node
import math_utils



F = Frame()
node1 = Node(coords = np.array([0, 0, 0]), F_x = 500*np.cos(40*np.pi/180), F_y = 500*np.sin(40*np.pi/180), u_z = 0, theta_x = 0, theta_y = 0, theta_z = 0) # node a
node2 = Node(coords = np.array([4, -4, 0]), F_x = 0, theta_z = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0)
elem1 = Element(node_list=[node1, node2], A = 20e3, E = 200, Iy = 0, Iz = 0, J = 1, nu = 0.3)

print(elem1)

def test_element_params():
    assert elem1.E == 200
    assert elem1.A == 20e3
    assert elem1.nu == 0.3
    assert elem1.Iy == 0
    assert elem1.Iz == 0
    assert elem1.J == 1

def test_element_length():
    assert elem1.L == np.sqrt(2*4**2)

def test_element_local_stiffness_mat():
    expected_local_stiffness_mat = math_utils.local_elastic_stiffness_matrix_3D_beam(elem1.E, elem1.nu, elem1.A, elem1.L, elem1.Iy, elem1.Iz, elem1.J)
    assert np.allclose(elem1.stiffness_mat(), expected_local_stiffness_mat)

def test_element_rotation_matrix():
    gamma = math_utils.rotation_matrix_3D(v_temp=elem1.local_z, *elem1.coord_list)
    Gamma = math_utils.transformation_matrix_3D(gamma)
    assert np.allclose(elem1.Gamma(), Gamma)

def test_element_global_stiffness_mat():
    gamma = math_utils.rotation_matrix_3D(v_temp=elem1.local_z, *elem1.coord_list)
    Gamma = math_utils.transformation_matrix_3D(gamma)
    expected_global_stiffness_mat = Gamma.T @ elem1.stiffness_mat() @ Gamma
    assert np.allclose(elem1.global_stiffness_mat(), expected_global_stiffness_mat)