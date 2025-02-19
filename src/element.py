import numpy
from frame import Frame

class Element(Frame):
    def __init__(self, **kwargs):
        # If shared_value has been defined (not None), use it.
        # Otherwise, use the value given by the user.
        if Element.E is not None:
            self.E = Element.E
        else:
            self.E = kwargs.get('E', 1.0)
        
        if Element.A is not None:
            self.A = Element.A
        else:
            self.A = kwargs.get('A', 1.0)
        
        if Element.L is not None:
            self.L = Element.L
        else:
            self.L = kwargs.get('L', 1.0)

        if Element.I is not None:
            self.I = Element.I
        else:
            self.I = kwargs.get('I', 1.0)


    def __repr__(self):
        return f"Element(variable={self.variable})"
