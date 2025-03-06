import numpy as np
import matplotlib.pyplot as plt

from element import Element
from node import Node


def plot_original(elem_list, fig, discretization_points: int = 100):
    ax = fig.add_subplot(111, projection='3d')
    for elem_ in elem_list:
        x_lin = np.linspace(elem_.node_list[0].coords[0], elem_.node_list[1].coords[0], discretization_points)
        y_lin = np.linspace(elem_.node_list[0].coords[1], elem_.node_list[1].coords[1], discretization_points)
        z_lin = np.linspace(elem_.node_list[0].coords[2], elem_.node_list[1].coords[2], discretization_points)
        ax.plot(x_lin, y_lin, z_lin, '--k', label='Original Shape')
    plt.show()

def plot_deformed(Frame, elem_list, shape_function, displacement_vector, scale: float = 1, discretization_points: int = 100):
    free_dofs = Frame.free_dofs
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for elem_ in elem_list:
        x_lin = np.linspace(elem_.node_list[0].coords[0], elem_.node_list[1].coords[0], discretization_points)
        y_lin = np.linspace(elem_.node_list[0].coords[1], elem_.node_list[1].coords[1], discretization_points)
        z_lin = np.linspace(elem_.node_list[0].coords[2], elem_.node_list[1].coords[2], discretization_points)
        ax.plot(x_lin, y_lin, z_lin, '--k')
    ax.plot(x_lin, y_lin, z_lin, '--k', label='Original Shape')
    deformed_shape_list = []
    for elem_ in elem_list:
        deformed_shape_array = shape_function.apply(displacement_vector * scale, elem_)
        # deformed_shape_array.Transpose(-1, 0, 1)
        deformed_shape_list.append(deformed_shape_array)
        discretization_points = len(deformed_shape_array)
        ax.scatter(deformed_shape_array[0, 0], deformed_shape_array[0, 1], deformed_shape_array[0, 2], c='r', s=10)
        ax.scatter(deformed_shape_array[-1, 0], deformed_shape_array[-1, 1], deformed_shape_array[-1, 2], c='r', s=10)
        x_lin = np.linspace(deformed_shape_array[0, 0], deformed_shape_array[-1, 0], discretization_points)
        y_lin = np.linspace(deformed_shape_array[0, 1], deformed_shape_array[-1, 1], discretization_points)
        ax.plot(x_lin, y_lin, deformed_shape_array[:, 2],'-b')
    ax.plot(x_lin, y_lin, deformed_shape_array[:, 2],'-b', label='Deformed Shape')
    # plt.scatter(deformed_shape_list[0][0, 0], deformed_shape_list[0][0, 1], deformed_shape_list[0][0, 2], c='r')
    # plt.scatter(deformed_shape_list[1][0, 0], deformed_shape_list[1][0, 1], deformed_shape_list[1][0, 2], c='b')
    # plt.scatter(deformed_shape_list[0][-1, 0], deformed_shape_list[0][-1, 1], deformed_shape_list[0][-1, 2], c='g')
    # plt.scatter(deformed_shape_list[1][-1, 0], deformed_shape_list[1][-1, 1], deformed_shape_list[1][-1, 2], c='y')
    deformed_shape = np.concatenate(deformed_shape_list)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x_lin, y_lin, deformed_shape[:, 2],'-b', label='Deformed Shape')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_mode_shape(Frame, elem_list, shape_function, eigenvector, scale: float = 1, discretization_points: int = 20):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    node_coords = np.concatenate([Frame.nodes[i].coords for i in range(len(Frame.nodes))]).reshape(-1, 3)
    
    # Plot original shape
    for elem_ in elem_list:
        x_lin = np.linspace(elem_.node_list[0].coords[0], elem_.node_list[1].coords[0], discretization_points)
        y_lin = np.linspace(elem_.node_list[0].coords[1], elem_.node_list[1].coords[1], discretization_points)
        z_lin = np.linspace(elem_.node_list[0].coords[2], elem_.node_list[1].coords[2], discretization_points)
        ax.plot(x_lin, y_lin, z_lin, '--k')
    ax.plot(x_lin, y_lin, z_lin, '--k', label='Original Shape')
    
    # Prepare eigenvector data
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
        
        # Get beam direction vector and length
        beam_vec = node1_coords - node0_coords
        beam_length = np.linalg.norm(beam_vec)
        beam_dir = beam_vec / beam_length
        
        # Create local coordinate system
        x_local = beam_dir
        # Choose appropriate reference for constructing local y and z
        if abs(np.dot(x_local, np.array([0, 0, 1]))) < 0.95:
            y_temp = np.cross(np.array([0, 0, 1]), x_local)
        else:
            y_temp = np.cross(np.array([0, 1, 0]), x_local)
        y_temp = y_temp / np.linalg.norm(y_temp)
        z_local = np.cross(x_local, y_temp)
        z_local = z_local / np.linalg.norm(z_local)
        y_local = np.cross(z_local, x_local)
        
        # Transformation matrix (from global to local)
        T = np.zeros((3, 3))
        T[0, :] = x_local
        T[1, :] = y_local
        T[2, :] = z_local
        
        # Transform displacements to local coordinates
        node0_disp_local = T @ node0_disp
        node1_disp_local = T @ node1_disp
        node0_rot_local = T @ node0_rot
        node1_rot_local = T @ node1_rot
        
        # Generate points along the beam in local coordinates
        s_values = np.linspace(0, 1, discretization_points)
        local_coords = np.zeros((discretization_points, 3))
        
        # Apply shape functions for all directions
        for i in range(3):
            # Linear shape functions for axial (along beam) displacement
            if i == 0:
                # Linear interpolation for x-direction (axial)
                N1 = 1 - s_values
                N2 = s_values
                local_coords[:, i] = s_values * beam_length + N1 * node0_disp_local[i] + N2 * node1_disp_local[i]
            else:
                # Hermite shape functions for transverse displacements (y and z)
                N1 = 1 - 3*s_values**2 + 2*s_values**3
                N2 = 3*s_values**2 - 2*s_values**3
                N3 = beam_length * (s_values - 2*s_values**2 + s_values**3)
                N4 = beam_length * (-s_values**2 + s_values**3)
                
                # For transverse directions, we need to map the rotation indices correctly
                # For y-direction (i=1), use rotation about z-axis (index 2)
                # For z-direction (i=2), use rotation about y-axis (index 1) with negative sign
                # rot_idx = 2 if i == 1 else 1
                # rot_sign = 1.0 if i == 1 else -1.0
                
                # Apply shape functions for transverse directions
                local_coords[:, i] = (
                    N1 * node0_disp_local[i] +
                    N2 * node1_disp_local[i] +
                    N3 * node0_rot_local[i-1] +  # Rotation about axis perpendicular to i direction
                    N4 * node1_rot_local[i-1]

                    # rot_sign * N3 * node0_rot_local[rot_idx] +
                    # rot_sign * N4 * node1_rot_local[rot_idx]
                )
        
        # Transform back to global coordinates and add original position
        global_coords = np.zeros((discretization_points, 3))
        for i in range(discretization_points):
            # Use transposed transformation matrix to go from local to global
            global_coords[i] = node0_coords + T.T @ local_coords[i]
        
        # Plot the deformed shape
        ax.plot(global_coords[:, 0], global_coords[:, 1], global_coords[:, 2], '-r')
        ax.scatter(global_coords[0, 0], global_coords[0, 1], global_coords[0, 2], c='r', s=10)
        ax.scatter(global_coords[-1, 0], global_coords[-1, 1], global_coords[-1, 2], c='r', s=10)
    
    # First element for legend
    ax.plot([], [], [], '-r', label='Mode Shape')
    # ax.set_xlim(-10, 20)
    # ax.set_ylim(-5, 30)
    # ax.set_zlim(0, 30)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Buckling Mode Shape')
    plt.show()

class HermiteShapeFunctions:
    
    def apply(self, displacement_vector: np.ndarray, elem: Element, discretization_points: int = 20):
        X_node_1 = elem.node_list[0].coords
        X_node_2 = elem.node_list[1].coords
        X_node_1_deformed = displacement_vector[0][:3]
        X_node_2_deformed = displacement_vector[1][:3]
        theta_node_1_deformed = displacement_vector[0][3:]
        theta_node_2_deformed = displacement_vector[1][3:]
        x_new = np.linspace(0, X_node_2_deformed - X_node_1_deformed, discretization_points)
        L_new = np.array(X_node_2_deformed - X_node_1_deformed) + np.array(np.finfo(float).eps)
        N1 = 1 - 3*(x_new/L_new)**2 + 2*(x_new/L_new)**3
        N2 = 3*(x_new/L_new)**2 - 2*(x_new/L_new)**3
        N3 = x_new*(1 - x_new/L_new)**2
        N4 = x_new * ((x_new/L_new)**2 - x_new/L_new)
        # deformed_shape = N1 * X_node_1_deformed[1] + N2 * X_node_2_deformed[1] + N3 * theta_node_1_deformed[-1] + N4 * theta_node_2_deformed[ -1]
        deformed_shape = N1 * X_node_1_deformed + N2 * X_node_2_deformed + N3 * theta_node_1_deformed + N4 * theta_node_2_deformed
        return x_new + X_node_1_deformed, deformed_shape