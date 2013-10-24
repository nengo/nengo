from .. import objects

class Network(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.objects = []
        self.make(*args, **kwargs)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            obj.name = self.name + '.' +  obj.name
            model.add(obj)
