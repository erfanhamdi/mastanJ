import numpy as np
import matplotlib.pyplot as plt

def plot_original_shape(elem_list, discretization_points: int = 20):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for elem_ in elem_list:
        x_lin = np.linspace(elem_.node_list[0].coords[0], elem_.node_list[1].coords[0], discretization_points)
        y_lin = np.linspace(elem_.node_list[0].coords[1], elem_.node_list[1].coords[1], discretization_points)
        z_lin = np.linspace(elem_.node_list[0].coords[2], elem_.node_list[1].coords[2], discretization_points)
        ax.plot(x_lin, y_lin, z_lin, '--k')
    ax.plot([], [], [], '--k', label='Original Shape')
    return fig, ax

def plot_mode_shape(Frame, elem_list, shape_function, eigenvector, scale: float = 1, discretization_points: int = 20):
    fig, ax = plot_original_shape(elem_list, discretization_points)
    free_dofs = Frame.free_dofs
    fixed_dofs = Frame.fixed_dofs
    eigenvector_array = np.zeros(len(free_dofs) + len(fixed_dofs))
    eigenvector_array[free_dofs] = eigenvector
    eigenvector_array[fixed_dofs] = 0
    eigenvector_array = eigenvector_array.reshape(-1, 6)
    
    for elem_ in elem_list:
        node_ids = [elem_.node_list[0].id, elem_.node_list[1].id]
        # Get node coordinates and eigenvector displacements
        node0_coords = elem_.node_list[0].coords
        node1_coords = elem_.node_list[1].coords
        node0_disp = eigenvector_array[node_ids[0], :3] * scale
        node1_disp = eigenvector_array[node_ids[1], :3] * scale
        node0_rot = eigenvector_array[node_ids[0], 3:] * scale
        node1_rot = eigenvector_array[node_ids[1], 3:] * scale
        node0_coords_deformed = node0_coords + node0_disp
        node1_coords_deformed = node1_coords + node1_disp
        
        beam_length = np.linalg.norm(node1_coords_deformed - node0_coords_deformed)
        T = elem_.Gamma()[:3, :3]
        
        # Transform displacements to local coordinates
        node0_disp_local = T @ node0_disp
        node1_disp_local = T @ node1_disp
        node0_rot_local = T @ node0_rot
        node1_rot_local = T @ node1_rot
        
        # Generate points along the beam in local coordinates
        x_values = np.linspace(0, beam_length, discretization_points)
        local_coords = np.zeros((discretization_points, 3))
        
        # Apply shape functions for all directions
        for i in range(3):
            # Linear shape functions for axial (along beam) displacement
            if i == 0:
                # Linear interpolation for x-direction (axial)
                N1 = 1 - x_values/beam_length
                N2 = x_values/beam_length
                local_coords[:, i] = x_values + N1 * node0_disp_local[i] + N2 * node1_disp_local[i]
            else:
                N1, N2, N3, N4 = shape_function.apply(x_values, beam_length)
                local_coords[:, i] = (
                    N1 * node0_disp_local[i] +
                    N2 * node1_disp_local[i] +
                    N3 * node0_rot_local[i-1] + 
                    N4 * node1_rot_local[i-1]
                )
        
        # Transform back to global 
        global_coords = np.zeros((discretization_points, 3))
        for i in range(discretization_points):
            global_coords[i] = node0_coords + T.T @ local_coords[i]
        
        # Plot the deformed shape
        ax.plot(global_coords[:, 0], global_coords[:, 1], global_coords[:, 2], '-r')
        ax.scatter(global_coords[0, 0], global_coords[0, 1], global_coords[0, 2], c='r', s=10)
        ax.scatter(global_coords[-1, 0], global_coords[-1, 1], global_coords[-1, 2], c='r', s=10)
    
    ax.plot([], [], [], '-r', label='Mode Shape')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Buckling Mode Shape')
    plt.show()

class HermiteShapeFunctions:
    def apply(self, x_values, beam_length):
        N1 = 1 - 3*(x_values/beam_length)**2 + 2*(x_values/beam_length)**3
        N2 = 3*(x_values/beam_length)** 2 - 2*(x_values/beam_length)**3
        N3 = x_values*(1 - x_values/beam_length)**2
        N4 = x_values * ((x_values/beam_length)**2 - x_values/beam_length)
        return N1, N2, N3, N4