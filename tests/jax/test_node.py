import pytest
import jax.numpy as jnp
from src.jax.node import Node

class TestJaxNode:
    def test_node_creation_basic(self):
        # Test with different coordinate input types
        node1 = Node(coords=(0., 0., 0.))
        node2 = Node(coords=[0., 0., 0.])
        node3 = Node(coords=jnp.array([0., 0., 0.]))

        for node in [node1, node2, node3]:
            assert node.id == 0
            assert all(bc is None for bc in node.bc)
            assert all(load is None for load in node.load)
            assert node.free_dofs == list(range(6))  # All DOFs are free by default
            assert node.fixed_dofs == []
            assert isinstance(node.coords, jnp.ndarray)

    def test_invalid_coords(self):
        # Test invalid coordinate type
        with pytest.raises(TypeError, match="coords must be a tuple, list or array of 3 numbers"):
            Node(coords="invalid")
            
        # Test invalid coordinate length
        with pytest.raises(ValueError, match="coords must contain exactly 3 values"):
            Node(coords=(1, 2))

    def test_invalid_parameters(self):
        # Test invalid parameter name
        with pytest.raises(ValueError, match="Unknown parameter: invalid_param"):
            Node(coords=(0., 0., 0.), invalid_param=42)
            
        # Test invalid boundary condition type
        with pytest.raises(TypeError, match="Boundary condition u_x must be numeric or None"):
            Node(coords=(0., 0., 0.), u_x="fixed")
            
        # Test invalid load type
        with pytest.raises(TypeError, match="Load F_x must be numeric or None"):
            Node(coords=(0., 0., 0.), F_x="1000N")

    def test_node_creation_with_boundary_conditions(self):
        node = Node(
            coords=jnp.array([1., 2., 3.]),
            u_x=0.,
            u_y=0.,
            u_z=0.,
            theta_x=None,
            theta_y=None,
            theta_z=None
        )
        assert jnp.array_equal(node.coords, jnp.array([1., 2., 3.]))
        assert jnp.array_equal(node.u_x, jnp.array(0.))
        assert jnp.array_equal(node.u_y, jnp.array(0.))
        assert jnp.array_equal(node.u_z, jnp.array(0.))
        assert node.theta_x is None
        assert node.theta_y is None
        assert node.theta_z is None
        assert node.free_dofs == [3, 4, 5]  # Only rotational DOFs are free
        assert node.fixed_dofs == [0, 1, 2]  # Translational DOFs are fixed

    def test_jax_array_properties(self):
        node = Node(
            coords=(1., 2., 3.),
            u_x=0.,
            F_y=1000.
        )
        assert isinstance(node.coords, jnp.ndarray)
        assert isinstance(node.u_x, jnp.ndarray)
        assert isinstance(node.F_y, jnp.ndarray) 