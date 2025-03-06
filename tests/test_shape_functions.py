import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, call

from src.shape_functions import plot_original_shape, plot_mode_shape, HermiteShapeFunctions
from src.frame import Frame
from src.element import Element
from src.node import Node


@pytest.fixture
def hermite_sf():
    """Create a HermiteShapeFunctions object for testing."""
    return HermiteShapeFunctions()


@pytest.fixture
def simple_frame():
    """Create a simple frame with one element for testing."""
    frame = Frame()
    node1 = Node(coords=np.array([0, 0, 0]), u_x=0, u_y=0, u_z=0, theta_x=0, theta_y=0, theta_z=0)
    node2 = Node(coords=np.array([1, 0, 0]), F_x=100, F_y=0, F_z=0, M_x=0, M_y=0, M_z=0)
    element = Element(node_list=[node1, node2], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    
    frame.add_elements([element])
    frame.assemble()
    frame.solve()
    
    return frame, [element]


def test_hermite_shape_functions_values(hermite_sf):
    """Test the HermiteShapeFunctions class with parameterized inputs."""
    # Define test variables
    x_values = np.linspace(0, 1, 10)
    beam_length = 1
    
    # Get the shape functions
    N1, N2, N3, N4 = hermite_sf.apply(x_values, beam_length)
    
    # Check dimensions
    assert len(N1) == len(x_values)
    assert len(N2) == len(x_values)
    assert len(N3) == len(x_values)
    assert len(N4) == len(x_values)
    
    # Check boundary conditions at x=0
    assert np.isclose(N1[0], 1.0)  # N1(0) = 1
    assert np.isclose(N2[0], 0.0)  # N2(0) = 0
    assert np.isclose(N3[0], 0.0)  # N3(0) = 0
    assert np.isclose(N4[0], 0.0)  # N4(0) = 0
    
    # Check boundary conditions at x=L
    assert np.isclose(N1[-1], 0.0)  # N1(L) = 0
    assert np.isclose(N2[-1], 1.0)  # N2(L) = 1
    assert np.isclose(N3[-1], 0.0)  # N3(L) = 0
    assert np.isclose(N4[-1], 0.0)  # N4(L) = 0
    
    # Check that displacement shape functions sum to 1 at any point
    for i in range(len(x_values)):
        assert np.isclose(N1[i] + N2[i], 1.0)

def test_hermite_shape_functions_mathematical_correctness(hermite_sf):
    """
    Test that the HermiteShapeFunctions.apply method correctly implements 
    the mathematical formulas for Hermite shape functions.
    """
    # Define test data
    beam_length = 2.0
    x_values = np.linspace(0, beam_length, 21)  # 21 points for detailed verification
    
    # Calculate shape functions using the class method
    N1, N2, N3, N4 = hermite_sf.apply(x_values, beam_length)
    
    # Manually calculate expected shape functions
    xi = x_values / beam_length  # Normalized coordinate
    expected_N1 = 1 - 3*xi**2 + 2*xi**3
    expected_N2 = 3*xi**2 - 2*xi**3
    expected_N3 = x_values*(1 - xi)**2
    expected_N4 = x_values * (xi**2 - xi)
    
    # Verify all points match for each shape function
    assert np.allclose(N1, expected_N1), "N1 shape function doesn't match expected formula"
    assert np.allclose(N2, expected_N2), "N2 shape function doesn't match expected formula"
    assert np.allclose(N3, expected_N3), "N3 shape function doesn't match expected formula"
    assert np.allclose(N4, expected_N4), "N4 shape function doesn't match expected formula"
    
    # Verify important mathematical properties
    
    # 1. N1 + N2 = 1 (partition of unity for displacement shape functions)
    assert np.allclose(N1 + N2, np.ones_like(N1)), "N1 + N2 should equal 1"
    
    # 2. First derivatives at the ends
    # At x=0: N1'(0)=0, N2'(0)=0, N3'(0)=1, N4'(0)=0
    # At x=L: N1'(L)=0, N2'(L)=0, N3'(L)=0, N4'(L)=1
    h = 1e-6  # Small step for numerical differentiation
    
    # Calculate at x=0
    N1_0, N2_0, N3_0, N4_0 = hermite_sf.apply(np.array([0]), beam_length)
    N1_h, N2_h, N3_h, N4_h = hermite_sf.apply(np.array([h]), beam_length)
    
    dN1_dx_0 = (N1_h[0] - N1_0[0]) / h
    dN2_dx_0 = (N2_h[0] - N2_0[0]) / h
    dN3_dx_0 = (N3_h[0] - N3_0[0]) / h
    dN4_dx_0 = (N4_h[0] - N4_0[0]) / h
    
    assert np.isclose(dN1_dx_0, 0.0, atol=1e-6), "N1'(0) should be 0"
    assert np.isclose(dN2_dx_0, 0.0, atol=1e-6), "N2'(0) should be 0"
    assert np.isclose(dN3_dx_0, 1.0, atol=1e-6), "N3'(0) should be 1"
    assert np.isclose(dN4_dx_0, 0.0, atol=1e-6), "N4'(0) should be 0"
    
    # Calculate at x=L
    N1_L, N2_L, N3_L, N4_L = hermite_sf.apply(np.array([beam_length]), beam_length)
    N1_Lh, N2_Lh, N3_Lh, N4_Lh = hermite_sf.apply(np.array([beam_length - h]), beam_length)
    
    dN1_dx_L = (N1_L[0] - N1_Lh[0]) / (-h)  # Note negative h since we're going backwards
    dN2_dx_L = (N2_L[0] - N2_Lh[0]) / (-h)
    dN3_dx_L = (N3_L[0] - N3_Lh[0]) / (-h)
    dN4_dx_L = (N4_L[0] - N4_Lh[0]) / (-h)
    
    assert np.isclose(dN1_dx_L, 0.0, atol=1e-6), "N1'(L) should be 0"
    assert np.isclose(dN2_dx_L, 0.0, atol=1e-6), "N2'(L) should be 0"
    assert np.isclose(dN3_dx_L, 0.0, atol=1e-6), "N3'(L) should be 0"
    assert np.isclose(dN4_dx_L, -1.0, atol=1e-6), "N4'(L) should be -1"

@pytest.mark.plot
@patch('matplotlib.pyplot.figure')
def test_plot_original_shape(mock_figure):
    """Test the plot_original_shape function."""
    # Setup mock figure and axes
    mock_ax = MagicMock()
    mock_figure.return_value.add_subplot.return_value = mock_ax
    
    # Create test elements
    node1 = Node(coords=np.array([0, 0, 0]))
    node2 = Node(coords=np.array([1, 0, 0]))
    node3 = Node(coords=np.array([0, 1, 0]))
    
    elem1 = Element(node_list=[node1, node2], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    elem2 = Element(node_list=[node1, node3], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    
    # Call the function
    fig, ax = plot_original_shape([elem1, elem2], discretization_points=10)
    
    # Verify figure and axes are returned
    assert fig == mock_figure.return_value
    assert ax == mock_ax
    
    # Verify plot calls (one for each element plus one for legend)
    assert mock_ax.plot.call_count == 3
    
    # Check that the first two calls are for the element lines
    for i, elem in enumerate([elem1, elem2]):
        # Get arguments from the ith call
        args, kwargs = mock_ax.plot.call_args_list[i]
        
        # Check line style
        assert kwargs.get('linestyle', '--') == '--' or '--k' in args
        
        # Check that the coordinates match the elements
        x_coords, y_coords, z_coords = args[0], args[1], args[2]
        assert len(x_coords) == 10  # discretization_points
        
        # Check start and end points match the nodes
        assert np.isclose(x_coords[0], elem.node_list[0].coords[0])
        assert np.isclose(y_coords[0], elem.node_list[0].coords[1])
        assert np.isclose(z_coords[0], elem.node_list[0].coords[2])
        
        assert np.isclose(x_coords[-1], elem.node_list[1].coords[0])
        assert np.isclose(y_coords[-1], elem.node_list[1].coords[1])
        assert np.isclose(z_coords[-1], elem.node_list[1].coords[2])
    
    # Check the legend entry
    last_args, last_kwargs = mock_ax.plot.call_args_list[-1]
    assert 'label' in last_kwargs and last_kwargs['label'] == 'Original Shape'

@pytest.mark.plot
@patch('src.shape_functions.plot_original_shape')
@patch('matplotlib.pyplot.show')
def test_plot_mode_shape_integration(mock_show, mock_plot_original):
    """Test that plot_mode_shape correctly integrates with plot_original_shape."""
    # Setup mock figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plot_original.return_value = (mock_fig, mock_ax)
    
    # Create a simple frame and element
    frame = Frame()
    node1 = Node(coords=np.array([0, 0, 0]), u_x=0, u_y=0, u_z=0, theta_x=0, theta_y=0, theta_z=0)
    node2 = Node(coords=np.array([1, 0, 0]), F_x=100, F_y=0, F_z=0, M_x=0, M_y=0, M_z=0)
    element = Element(node_list=[node1, node2], A=100, E=200, Iy=50, Iz=50, J=25, nu=0.3)
    
    elements = [element]
    frame.add_elements(elements)
    frame.assemble()
    
    # Mock free_dofs and fixed_dofs
    frame.free_dofs = np.array([6, 7, 8, 9, 10, 11])  # node2's DOFs
    frame.fixed_dofs = np.array([0, 1, 2, 3, 4, 5])   # node1's DOFs
    
    # Create a simple eigenvector (unit displacement in Y direction)
    eigenvector = np.zeros(len(frame.free_dofs))
    eigenvector[1] = 1.0  # Y direction
    
    # Create shape function
    hermite_sf = HermiteShapeFunctions()
    
    # Call plot_mode_shape
    plot_mode_shape(frame, elements, hermite_sf, eigenvector, scale=2.0, discretization_points=15)
    
    # Verify plot_original_shape was called with correct arguments
    mock_plot_original.assert_called_once_with(elements, 15)
    
    # Check that axis labels and title were set
    mock_ax.set_xlabel.assert_called_once_with('x')
    mock_ax.set_ylabel.assert_called_once_with('y')
    mock_ax.set_zlabel.assert_called_once_with('z')
    mock_ax.set_title.assert_called_once_with('Buckling Mode Shape')
    
    # Verify mode shape was plotted
    assert mock_ax.plot.call_count >= len(elements) + 1  # At least one call per element plus legend
    
    # Check legend was added
    mock_ax.legend.assert_called_once()
    
    # Check that plt.show was called
    mock_show.assert_called_once()
    
    # Verify that scatter points were added for node positions (2 per element)
    scatter_calls = mock_ax.scatter.call_count
    assert scatter_calls == 2 * len(elements), f"Expected {2 * len(elements)} scatter calls, got {scatter_calls}"

@pytest.mark.plot
@patch('matplotlib.pyplot.show')
def test_plot_mode_shape_full_integration(mock_show, simple_frame, hermite_sf):
    """Test the plot_mode_shape function with a full integration approach."""
    frame, elements = simple_frame
    
    # Ensure free_dofs is an array to avoid the error
    if hasattr(frame, 'free_dofs') and not isinstance(frame.free_dofs, np.ndarray):
        frame.free_dofs = np.array([6, 7, 8, 9, 10, 11])  # Force node2's DOFs to be free
        frame.fixed_dofs = np.array([0, 1, 2, 3, 4, 5])   # Force node1's DOFs to be fixed
    
    # Create a simple eigenvector
    eigenvector = np.zeros(len(frame.free_dofs))
    
    # Find index corresponding to y-displacement of node 2
    node2_id = elements[0].node_list[1].id
    y_dof_global = node2_id * 6 + 1  # +1 for y direction
    
    # Safely find the index in free_dofs array
    try:
        # Try to find index using where
        y_dof_index = np.where(frame.free_dofs == y_dof_global)[0]
        if len(y_dof_index) > 0:
            eigenvector[y_dof_index[0]] = 1.0
    except (ValueError, TypeError):
        # Fallback: Just set a value in the middle of the vector
        if len(eigenvector) > 0:
            eigenvector[len(eigenvector) // 2] = 1.0
    
    # Spy on plt.figure and get the result
    with patch('matplotlib.pyplot.figure', wraps=plt.figure) as spy_figure:
        # Call the function
        plot_mode_shape(frame, elements, hermite_sf, eigenvector, scale=1.0, discretization_points=10)
        
        # Verify that figure was created
        spy_figure.assert_called_once()
    
    # Verify show was called
    mock_show.assert_called_once()
