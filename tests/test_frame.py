import numpy as np
import pytest
import scipy.linalg

from src.frame import Frame
from src.element import Element
from src.node import Node
import src.math_utils as math_utils
from src.math_utils import check_unit_vector, check_parallel

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

def test_assemble_geometric():
    """Test the assemble_geometric method of the Frame class."""
    # Create a simple frame with known properties for testing
    test_frame = Frame()
    test_node1 = Node(coords=np.array([0, 0, 0]), u_x=0, u_y=0, u_z=0, theta_x=0, theta_y=0, theta_z=0)
    test_node2 = Node(coords=np.array([1, 0, 0]), F_x=100, F_y=0, F_z=0, M_x=0, M_y=0, M_z=0)
    test_elem = Element(node_list=[test_node1, test_node2], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    
    # Add element and assemble stiffness matrix
    test_frame.add_elements([test_elem])
    test_frame.assemble()
    
    # Solve to get displacement vector
    test_frame.solve()
    
    # The frame should have the Delta attribute after solving
    assert hasattr(test_frame, 'Delta'), "Frame should have Delta attribute after solving"
    
    # Store the number of DOFs before assembling geometric matrix
    num_dofs = test_frame.number_of_dofs
    
    # Call the method being tested
    test_frame.assemble_geometric()
    
    # Verify that K_g has been created with correct dimensions
    assert hasattr(test_frame, 'K_g'), "Frame should have K_g attribute after assemble_geometric"
    assert test_frame.K_g.shape == (num_dofs, num_dofs), f"K_g shape should be ({num_dofs}, {num_dofs})"
    
    # Verify that the partitioned matrices exist
    assert hasattr(test_frame, 'K_ff_g'), "Frame should have K_ff_g attribute after assemble_geometric"
    assert hasattr(test_frame, 'K_fs_g'), "Frame should have K_fs_g attribute after assemble_geometric"
    assert hasattr(test_frame, 'K_sf_g'), "Frame should have K_sf_g attribute after assemble_geometric"
    assert hasattr(test_frame, 'K_ss_g'), "Frame should have K_ss_g attribute after assemble_geometric"
    
    # Check that at least one of the matrices has non-zero values (showing it's been assembled)
    assert np.any(test_frame.K_g != 0), "K_g should have non-zero values after assembly"

def test_partition_geometric():
    """Test the partition_geometric method of the Frame class."""
    # Create a test frame with a simple structure
    test_frame = Frame()
    test_node1 = Node(coords=np.array([0, 0, 0]), u_x=0, u_y=0, u_z=0, theta_x=0, theta_y=0, theta_z=0)
    test_node2 = Node(coords=np.array([1, 0, 0]), F_x=100, F_y=0, F_z=0, M_x=0, M_y=0, M_z=0)
    test_elem = Element(node_list=[test_node1, test_node2], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    
    # Add element and assemble
    test_frame.add_elements([test_elem])
    test_frame.assemble()
    test_frame.solve()
    
    # Create a known K_g matrix for testing
    num_dofs = test_frame.number_of_dofs
    test_K_g = np.ones((num_dofs, num_dofs))
    np.fill_diagonal(test_K_g, 2.0)  # Add unique values on diagonal
    
    # Set the K_g matrix and call partition_geometric
    test_frame.K_g = test_K_g.copy()
    test_frame.partition_geometric()
    
    # Get the free and fixed DOFs
    free_dofs = test_frame.free_dofs
    fixed_dofs = test_frame.fixed_dofs
    
    # Verify the partitioned matrices
    assert np.array_equal(test_frame.K_ff_g, test_K_g[np.ix_(free_dofs, free_dofs)]), "K_ff_g doesn't match expected partition"
    assert np.array_equal(test_frame.K_fs_g, test_K_g[np.ix_(free_dofs, fixed_dofs)]), "K_fs_g doesn't match expected partition"
    assert np.array_equal(test_frame.K_sf_g, test_K_g[np.ix_(fixed_dofs, free_dofs)]), "K_sf_g doesn't match expected partition"
    assert np.array_equal(test_frame.K_ss_g, test_K_g[np.ix_(fixed_dofs, fixed_dofs)]), "K_ss_g doesn't match expected partition"

def test_eigenvalue_analysis():
    """Test the eigenvalue_analysis method of the Frame class."""
    # Create a test frame with a simple structure
    test_frame = Frame()
    test_node1 = Node(coords=np.array([0, 0, 0]), u_x=0, u_y=0, u_z=0, theta_x=0, theta_y=0, theta_z=0)
    test_node2 = Node(coords=np.array([1, 0, 0]), F_x=100, F_y=0, F_z=0, M_x=0, M_y=0, M_z=0)
    test_elem = Element(node_list=[test_node1, test_node2], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    
    # Add element and assemble
    test_frame.add_elements([test_elem])
    test_frame.assemble()
    test_frame.solve()
    
    # Create simple matrices for K_ff and K_ff_g for testing
    # We'll set these directly to avoid complexity in the test
    free_dofs = test_frame.free_dofs
    n_free = len(free_dofs)
    
    # Create a simple positive definite matrix for K_ff
    test_K_ff = np.eye(n_free) * 10
    # Create a simple negative definite matrix for K_ff_g
    test_K_ff_g = -np.eye(n_free) * 5
    
    # Set the matrices directly
    test_frame.K_ff = test_K_ff.copy()
    test_frame.K_ff_g = test_K_ff_g.copy()
    
    # Call eigenvalue_analysis
    eigenvalues, eigenvectors = test_frame.eigenvalue_analysis()
    
    # Calculate expected eigenvalues and eigenvectors using scipy directly
    expected_eigenvalues, expected_eigenvectors = scipy.linalg.eig(test_K_ff, -test_K_ff_g)
    
    # Verify the eigenvalues and eigenvectors
    assert len(eigenvalues) == n_free, f"Expected {n_free} eigenvalues, got {len(eigenvalues)}"
    assert eigenvectors.shape == (n_free, n_free), f"Expected eigenvector shape {(n_free, n_free)}, got {eigenvectors.shape}"
    
    # Eigenvalues should match those calculated directly
    assert np.allclose(np.sort(eigenvalues), np.sort(expected_eigenvalues)), "Eigenvalues don't match expected values"
    
    # Check that the eigenvectors are valid (we don't check exact equality because eigenvectors might differ by a constant factor)
    for i in range(n_free):
        # Select the ith eigenvector and eigenvalue
        v = eigenvectors[:, i]
        lambda_val = eigenvalues[i]
        
        # Check that K_ff·v = λ·(-K_ff_g)·v
        lhs = test_K_ff @ v
        rhs = lambda_val * (-test_K_ff_g) @ v
        
        # The vectors should be proportional with ratio lambda
        assert np.allclose(lhs, rhs), f"Eigenvector {i} doesn't satisfy the eigenvalue equation"

def test_check_unit_vector():
    """Test the check_unit_vector function in math_utils."""
    # Test with a unit vector (should pass)
    unit_vectors = [
        np.array([1.0, 0.0, 0.0]),  # Unit vector along x
        np.array([0.0, 1.0, 0.0]),  # Unit vector along y
        np.array([0.0, 0.0, 1.0]),  # Unit vector along z
        np.array([0.7071, 0.7071, 0.0]),  # Normalized vector at 45 degrees in xy plane
        np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])  # Normalized vector along [1,1,1]
    ]
    
    # All of these should pass without raising an exception
    for vec in unit_vectors:
        check_unit_vector(vec)  # Should not raise any exception
    
    # Test with non-unit vectors (should raise ValueError)
    non_unit_vectors = [
        np.array([2.0, 0.0, 0.0]),  # Length 2
        np.array([0.5, 0.0, 0.0]),  # Length 0.5
        np.array([1.0, 1.0, 1.0]),  # Length sqrt(3)
        np.array([0.0, 0.0, 0.0])   # Zero vector
    ]
    
    # All of these should raise ValueError
    for vec in non_unit_vectors:
        with pytest.raises(ValueError, match="Expected a unit vector for reference vector."):
            check_unit_vector(vec)

def test_check_parallel():
    """Test the check_parallel function in math_utils."""
    # Test with vectors that are NOT parallel (should pass)
    non_parallel_pairs = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # x and y axes
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # y and z axes
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # x and z axes
        (np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # xy plane and z axis
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, -1.0, 0.0]))  # random non-parallel vectors
    ]
    
    # All of these should pass without raising an exception
    for vec1, vec2 in non_parallel_pairs:
        check_parallel(vec1, vec2)  # Should not raise any exception
    
    # Test with vectors that ARE parallel (should raise ValueError)
    parallel_pairs = [
        (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])),  # both along x axis
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, -3.0, 0.0])),  # both along y axis (opposite directions)
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5])),  # both along z axis
        (np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])),  # both along same diagonal
        (np.array([1.0, 1.0, 1.0]), np.array([-0.5, -0.5, -0.5]))  # both along same diagonal (opposite directions)
    ]
    
    # All of these should raise ValueError
    for vec1, vec2 in parallel_pairs:
        with pytest.raises(ValueError, match="Reference vector is parallel to beam axis."):
            check_parallel(vec1, vec2)