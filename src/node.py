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
        self.D = kwargs.get('D', None)
        self.F = kwargs.get('F', None)
        self.M = kwargs.get('M', None)