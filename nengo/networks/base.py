from .. import objects

class Network(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name

        self.objects = []

        self.make(name, *args, **kwargs)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def connect_to(self, post, **kwargs):
        raise NotImplementedError("Implement for default output.")

    @property
    def input(self):
        raise NotImplementedError("Implement for default input.")

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            obj.name = self.name + '.' +  obj.name
            model.add(obj)
