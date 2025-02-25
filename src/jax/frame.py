import jax.numpy as np
import jax.lax as lax
from jax import jit, vmap
from functools import partial
from typing import List, Dict, Optional, Tuple
from .element import Element


class Frame:
    def __init__(self):
            self.elems = []
            self.nodes = []
            self.free_dofs = []
            self.fixed_dofs = []
            self.K = None
            self.F = None
            self.Delta = None
            self.number_of_dofs = 0
            self.K_ff = None
            self.K_fs = None
            self.K_sf = None
            self.K_ss = None
            self.dofs_array = None
            self.Delta_f = None

    def add_elements(self, elems):
        self.elems = elems
        self.nodes = []
        
        # Process all nodes at once
        all_nodes = []
        for elem in elems:
            all_nodes.extend(elem.node_list)
            
        # Deduplicate nodes efficiently
        unique_nodes = []
        node_set = set()
        for node in all_nodes:
            node_hash = hash(node)
            if node_hash not in node_set:
                unique_nodes.append(node)
                node_set.add(node_hash)
        
        self.nodes = unique_nodes
        
        # Set IDs in one pass
        for elem_id, elem in enumerate(elems):
            elem.id = elem_id
        
        for node_id, node in enumerate(self.nodes):
            node.id = node_id
            
        self.number_of_dofs = 6 * len(self.nodes)

    def num_dofs(self):
        return 6 * len(self.nodes)

    def assemble(self):
        # Pre-compute all stiffness matrices
        elem_stiffness_matrices = []
        node_indices = []
        
        for elem in self.elems:
            elem_stiffness_matrices.append(elem.global_stiffness_mat())
            node_indices.append((elem.node_list[0].id, elem.node_list[1].id))
        
        # Initialize K
        self.K = np.zeros((self.number_of_dofs, self.number_of_dofs))
        
        # Assemble K manually without JIT
        for i, small_K in enumerate(elem_stiffness_matrices):
            node_1_id, node_2_id = node_indices[i]
            
            start1 = 6 * node_1_id
            end1 = 6 * (node_1_id + 1)
            start2 = 6 * node_2_id
            end2 = 6 * (node_2_id + 1)
            
            # Update all four blocks
            self.K = self.K.at[start1:end1, start1:end1].add(small_K[:6, :6])
            self.K = self.K.at[start1:end1, start2:end2].add(small_K[:6, 6:])
            self.K = self.K.at[start2:end2, start1:end1].add(small_K[6:, :6])
            self.K = self.K.at[start2:end2, start2:end2].add(small_K[6:, 6:])
        
        # Initialize F and Delta
        self.F = np.zeros(self.number_of_dofs)
        self.Delta = np.zeros(self.number_of_dofs)
        
        # Partition the system
        self.partition()
    
    def partition(self):
        # Collect all DOFs and loads efficiently
        free_dofs_list = []
        fixed_dofs_list = []
        loads = []
        
        for node in self.nodes:
            node_id = node.id
            node_offset = 6 * node_id
            
            # Add node's free and fixed DOFs with offset
            free_dofs_list.extend([i + node_offset for i in node.free_dofs])
            fixed_dofs_list.extend([i + node_offset for i in node.fixed_dofs])
            
            # Collect loads
            loads.append(node.load)
        
        # Convert to JAX arrays
        self.free_dofs = np.array(free_dofs_list, dtype=np.int32)
        self.fixed_dofs = np.array(fixed_dofs_list, dtype=np.int32)
        
        # Extract submatrices using JAX indexing
        self.K_ff = self.K[self.free_dofs][:, self.free_dofs]
        self.K_fs = self.K[self.free_dofs][:, self.fixed_dofs]
        self.K_sf = self.K[self.fixed_dofs][:, self.free_dofs]
        self.K_ss = self.K[self.fixed_dofs][:, self.fixed_dofs]
        
        # Update forces
        f_list = []
        for node in self.nodes:
            f_list.extend(node.load)
        
        f_array = np.array(f_list)
        if len(f_array) > 0:
            self.F = self.F.at[self.free_dofs].set(f_array[self.free_dofs])
    
    @partial(jit, static_argnums=(0,))
    def _solve_system(self, K_ff, F_free, K_sf):
        """JIT-compiled system solver for better performance"""
        # Solve for displacements
        delta_free = np.linalg.solve(K_ff, F_free)
        
        # Calculate reactions
        F_fixed = K_sf @ delta_free
        
        return delta_free, F_fixed
    
    def solve(self):
        # Use the JIT-compiled solver
        delta_free, F_fixed = self._solve_system(
            self.K_ff, 
            self.F[self.free_dofs],
            self.K_sf
        )
        
        # Update solution arrays
        self.Delta = self.Delta.at[self.free_dofs].set(delta_free)
        self.F = self.F.at[self.fixed_dofs].set(F_fixed)
        
        return self.Delta, self.F