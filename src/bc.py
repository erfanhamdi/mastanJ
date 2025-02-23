class BC:
    def __init__(self, **kwargs):
        self.u_x = kwargs.get('u_x', None)
        self.F_x = kwargs.get('F_x', None)
        self.check_bc(self.u_x, self.F_x)
        self.u_y = kwargs.get('u_y', None)
        self.F_y = kwargs.get('F_y', None)
        self.check_bc(self.u_y, self.F_y)
        self.u_z = kwargs.get('u_z', None)
        self.F_z = kwargs.get('F_z', None)
        self.check_bc(self.u_z, self.F_z)
        self.theta_x = kwargs.get('theta_x', None)
        self.M_x = kwargs.get('M_x', None)
        self.check_bc(self.theta_x, self.M_x)
        self.theta_y = kwargs.get('theta_y', None)
        self.M_y = kwargs.get('M_y', None)
        self.check_bc(self.theta_y, self.M_y)
        self.theta_z = kwargs.get('theta_z', None)
        self.M_z = kwargs.get('M_z', None)
        self.check_bc(self.theta_z, self.M_z)
    
    def check_bc(self, u, F):
        # make sure at least one is not None
        if u is None and F is None:
            raise ValueError("Either u or F must be provided")
    
    def __call__(self):
        return self.u_x, self.u_y, self.u_z, self.theta_x, self.theta_y, self.theta_z, self.F_x, self.F_y, self.F_z, self.M_x, self.M_y, self.M_z