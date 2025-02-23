import numpy
from frame import Frame
from math_utils import local_elastic_stiffness_matrix_3D_beam, rotation_matrix_3D, transformation_matrix_3D

class Element(Frame):
    _counter = 0
    def __init__(self, node_list, **kwargs):
        self.node_list = node_list
        self.coord_list = [node_list[0].coords[0], node_list[0].coords[1], node_list[0].coords[2], node_list[1].coords[0], node_list[1].coords[1], node_list[1].coords[2]]
        id = kwargs.get('id', None)
        if id is not None:
            self.id = id
        else:
            self.id = Element._counter
            Element._counter += 1
        
        if Element.E is not None:
            self.E = Element.E
        else:
            self.E = kwargs.get('E', 1.0)

        if Element.A is not None:
            self.A = Element.A
        else:
            self.A = kwargs.get('A', 1.0)
        
        self.L = numpy.linalg.norm(node_list[1].coords - node_list[0].coords)
        
        if Element.Iy is not None:
            self.Iy = Element.Iy
        else:
            self.Iy = kwargs.get('Iy', 1.0)

        if Element.Iz is not None:
            self.Iz = Element.Iz
        else:
            self.Iz = kwargs.get('Iz', 1.0)

        if Element.J is not None:
            self.J = Element.J
        else:
            self.J = kwargs.get('J', 1.0)
        
        if Element.nu is not None:
            self.nu = Element.nu
        else:
            self.nu = kwargs.get('nu', 0.3)
    
    def __repr__(self):
        # prints the element's properties
        return f"Element id: {self.id} (E={self.E}), A={self.A}, L={self.L}, Iy={self.Iy}, Iz={self.Iz}, J={self.J}, nu={self.nu}"
    
    def stiffness_mat(self,):
        return local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J)
    
    def Gamma(self,):
        gamma = rotation_matrix_3D(*self.coord_list)
        return transformation_matrix_3D(gamma)
    
    def global_stiffness_mat(self,):
        return self.Gamma().T @ self.stiffness_mat() @ self.Gamma()
        

# Example usage:

# Case 1: User defines a shared variable in Frame
# Frame.set_cross_section(E = 100, A = 200, L = 300, I = 400)
# elem1 = Element(E = 10)
# print(elem1)  # Output: Element(variable=100)
# Case 2: The shared variable is not set, so each Element gets its own value
# Frame.shared_value = None  # Unset the shared variable
# elem3 = Element(E = 30)
# elem4 = Element(A = 40)
# elem5 = Element(L = 50, id = 5)
# print(elem3)  # Output: Element(variable=30)
# print(elem4)  
# print(elem5)  # Output: Element(variable=50)