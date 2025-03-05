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
    
    free_dofs = Frame.free_dofs
    fixed_dofs = Frame.fixed_dofs
    eigenvector_array = np.zeros(len(free_dofs) + len(fixed_dofs))
    eigenvector_array[free_dofs] = eigenvector
    eigenvector_array[fixed_dofs] = 0
    # eigenvector_array = elem_list[0].Gamma() @ eigenvector_array
    eigenvector_array = eigenvector_array.reshape(-1, 6)
    eigen_vector_xyz = eigenvector_array[:, :3]
    deformed_coords = node_coords + eigen_vector_xyz * scale
    deformed_theta = eigenvector_array[:, 3:] * scale
    deformed_array = np.concatenate([deformed_coords, deformed_theta], axis=1)
    
    for elem_ in elem_list:
        node_ids = [elem_.node_list[0].id, elem_.node_list[1].id]
        # deformed_array_ = elem_.Gamma().T @ deformed_array[node_ids].reshape(-1)
        # deformed_array_ = deformed_array_.reshape(-1, 6)
        x_new, mode_shape_array = shape_function.apply(deformed_array[node_ids], elem_, discretization_points)
        
        # Determine beam orientation
        is_straight_beam = np.allclose(elem_.node_list[0].coords[1], elem_.node_list[1].coords[1])
        
        # Plot differently based on beam orientation
        if is_straight_beam:
            ax.plot(x_new[:, 0], mode_shape_array[:, 1], x_new[:, 2], '-r',  label='Mode Shape')
        else:
            ax.plot(x_new[:, 0], x_new[:, 1], mode_shape_array[:, 2], '-r',  label='Mode Shape')
        
        ax.scatter(x_new[0, 0], x_new[0, 1], x_new[0, 2], c='r', s=10)
        ax.scatter(x_new[-1, 0], x_new[-1, 1], x_new[-1, 2], c='r', s=10)
        # ax.plot(x_new[:, 0], x_new[:, 1], mode_shape_array[:, 2], '-r', label='Mode Shape')
    ax.set_zlim(-25, 15)
    # ax.legend()
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