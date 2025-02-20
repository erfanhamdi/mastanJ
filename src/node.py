import numpy as np

class Node:
    _counter = 0
    def __init__(self, coords, **kwargs):
        self.coords = coords
        self.id = kwargs.get('id', None)
        if self.id is not None:
            self.id = self.id
        else:
            self.id = self._counter
            Node._counter += 1
        self.u_x = kwargs.get('u_x', None)
        self.u_y = kwargs.get('u_y', None)
        self.u_z = kwargs.get('u_z', None)
        self.theta_x = kwargs.get('theta_x', None)
        self.theta_y = kwargs.get('theta_y', None)
        self.theta_z = kwargs.get('theta_z', None)

        self.F_x = kwargs.get('F_x', None)
        self.F_y = kwargs.get('F_y', None)
        self.F_z = kwargs.get('F_z', None)
        self.M_x = kwargs.get('M_x', None)
        self.M_y = kwargs.get('M_y', None)
        self.M_z = kwargs.get('M_z', None)

        self.bc = [self.u_x, self.u_y, self.u_z, self.theta_x, self.theta_y, self.theta_z]
        self.load = [self.F_x, self.F_y, self.F_z, self.M_x, self.M_y, self.M_z]
        self.free_dofs = [i for i, x in enumerate(self.bc) if x is None]
        self.fixed_dofs = [i for i, x in enumerate(self.bc) if x is not None]