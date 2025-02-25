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
        
    @partial(jit, static_argnums=(0,))
    def _add_element_stiffness(self, K, small_K, node_1_id, node_2_id):
        """JIT-compatible function to add element stiffness to global matrix"""
        start1 = node_1_id * 6
        start2 = node_2_id * 6
        
        # Update the four blocks using dynamic updates
        # Block 1,1
        for i in range(6):
            for j in range(6):
                K = K.at[start1 + i, start1 + j].add(small_K[i, j])
                
        # Block 1,2
        for i in range(6):
            for j in range(6):
                K = K.at[start1 + i, start2 + j].add(small_K[i, j + 6])
                
        # Block 2,1
        for i in range(6):
            for j in range(6):
                K = K.at[start2 + i, start1 + j].add(small_K[i + 6, j])
                
        # Block 2,2
        for i in range(6):
            for j in range(6):
                K = K.at[start2 + i, start2 + j].add(small_K[i + 6, j + 6])
                
        return K

    def assemble(self):
        # Initialize K
        self.K = np.zeros((self.number_of_dofs, self.number_of_dofs))
        
        # Process each element
        for elem in self.elems:
            small_K = elem.global_stiffness_mat()
            node_1_id = elem.node_list[0].id
            node_2_id = elem.node_list[1].id
            
            # Use JIT-compiled function to add element stiffness
            self.K = self._add_element_stiffness(
                self.K, 
                small_K, 
                np.array(node_1_id, dtype=np.int32), 
                np.array(node_2_id, dtype=np.int32)
            )
        
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