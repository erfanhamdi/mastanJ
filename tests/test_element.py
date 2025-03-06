import numpy as np

from src.frame import Frame
from src.element import Element
from src.node import Node
import math_utils
from math_utils import local_geometric_stiffness_matrix_3D_beam



F = Frame()
node1 = Node(coords = np.array([0, 0, 0]), F_x = 1, F_y = 1, F_z = 0, theta_x = 0, theta_y = 0, theta_z = 0) # node a
node2 = Node(coords = np.array([1, 0, 0]), u_x = 0, theta_z = 0, u_y = 0, u_z = 0, theta_x = 0, theta_y = 0)
elem1 = Element(node_list=[node1, node2], A = 20e3, E = 200, Iy = 1, Iz = 1, J = 1, nu = 0.3)

def test_element_params():
    assert elem1.E == 200
    assert elem1.A == 20e3
    assert elem1.nu == 0.3
    assert elem1.Iy == 1
    assert elem1.Iz == 1
    assert elem1.J == 1

def test_element_length():
    assert elem1.L == 1

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

def test_element_internal_force():
    F.add_elements([elem1])
    F.assemble()
    delta, F_rxn = F.solve()
    internal_force = elem1.internal_force(delta)
    assert np.allclose(internal_force, F_rxn)

def test_element_repr():
    repr_str = repr(elem1)
    expected_repr = f"Element id: E={elem1.E}, A={elem1.A}, L={elem1.L}, Iy={elem1.Iy}, Iz={elem1.Iz}, J={elem1.J}, nu={elem1.nu}"
    assert repr_str == expected_repr

def test_element_dof_list():
    dofs = elem1.dof_list()
    expected_dofs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    assert len(dofs) == 12, f"Expected 12 DOFs, got {len(dofs)}"
    assert dofs == expected_dofs, f"DOF list does not match expected: {dofs} vs {expected_dofs}"

def test_element_geometric_stiffness_mat():
    """Test the geometric_stiffness_mat method of the Element class."""
    # Create a force vector with known values
    # Format: [F_x1, F_y1, F_z1, M_x1, M_y1, M_z1, F_x2, F_y2, F_z2, M_x2, M_y2, M_z2]
    force_vector = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    
    # Extract the components needed for the geometric stiffness matrix
    # According to the method: F_list = [F[6], F[9], F[4], F[5], F[10], F[11]]
    # Which corresponds to: [F_x2, M_x2, M_y1, M_z1, M_y2, M_z2]
    F_x2 = force_vector[6]    # 70
    M_x2 = force_vector[9]    # 100
    M_y1 = force_vector[4]    # 50
    M_z1 = force_vector[5]    # 60
    M_y2 = force_vector[10]   # 110
    M_z2 = force_vector[11]   # 120
    
    # Calculate expected geometric stiffness matrix
    expected_k_g = local_geometric_stiffness_matrix_3D_beam(
        elem1.L, elem1.A, elem1.I_rho, F_x2, M_x2, M_y1, M_z1, M_y2, M_z2
    )
    
    # Call the method on the element
    elem1.geometric_stiffness_mat(force_vector)
    
    # Verify the result
    assert hasattr(elem1, 'k_g'), "Element should have a k_g attribute after calling geometric_stiffness_mat"
    assert elem1.k_g.shape == expected_k_g.shape, f"Shape mismatch: {elem1.k_g.shape} vs {expected_k_g.shape}"
    assert np.allclose(elem1.k_g, expected_k_g), "Geometric stiffness matrix does not match expected value"
    
    # Test with a different force vector
    force_vector2 = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115])
    expected_k_g2 = local_geometric_stiffness_matrix_3D_beam(
        elem1.L, elem1.A, elem1.I_rho, 
        force_vector2[6], force_vector2[9], force_vector2[4], 
        force_vector2[5], force_vector2[10], force_vector2[11]
    )
    
    elem1.geometric_stiffness_mat(force_vector2)
    assert np.allclose(elem1.k_g, expected_k_g2), "Geometric stiffness matrix does not match for second force vector"

def test_element_global_geometric_stiffness_mat():
    """Test the global_geometric_stiffness_mat method of the Element class."""
    # Create a sample local geometric stiffness matrix
    # We'll use a simple matrix for testing, but in reality it would be more complex
    sample_k_g = np.eye(12) * 2.5  # Simple diagonal matrix
    
    # Get the transformation matrix
    gamma = elem1.Gamma()
    
    # Calculate expected global geometric stiffness matrix
    expected_global_k_g = gamma.T @ sample_k_g @ gamma
    
    # Call the method and get the result
    result_global_k_g = elem1.global_geometric_stiffness_mat(sample_k_g)
    
    # Verify the result
    assert result_global_k_g.shape == (12, 12), f"Shape mismatch: {result_global_k_g.shape} vs (12, 12)"
    assert np.allclose(result_global_k_g, expected_global_k_g), "Global geometric stiffness matrix does not match expected value"
    
    # Test with a different matrix
    sample_k_g_2 = np.ones((12, 12)) * 0.5
    np.fill_diagonal(sample_k_g_2, 3.0)
    
    expected_global_k_g_2 = gamma.T @ sample_k_g_2 @ gamma
    result_global_k_g_2 = elem1.global_geometric_stiffness_mat(sample_k_g_2)
    
    assert np.allclose(result_global_k_g_2, expected_global_k_g_2), "Global geometric stiffness matrix does not match for second test matrix"