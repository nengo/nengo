import numpy as np

class FunctionOutput(object):
    
    def __init__(self, func, dimensions):
        """
        :param function func: the function carried out by this origin
        """

        self.func = func
        self.dimensions = dimensions

    def step(self, *args, **kwargs):
        self.value = np.asarray(self.func(*args, **kwargs))
        return self.value


class Output(object):
    def __init__(self, value):
        self.value = np.asarray(self.value)
        self.dimensions = self.value.shape

    def step(self):
        return self.value


