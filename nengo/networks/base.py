from .. import objects

class Network(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name

        self.objects = []
        self.connections_in = []
        self.connections_out = []

        self.make(*args, **kwargs)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def connect_to(self, post, **kwargs):
        raise NotImplementedError("Implement for default output.")

    # @property
    # def input(self):
    #     raise NotImplementedError("Implement for default input.")

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def probe(self, *args, **kwargs):
        raise NotImplementedError("Implement for default probe.")

    @property
    def signal(self):
        raise NotImplementedError("Implement for default input.")

    def add_to_model(self, model):
        if model.objs.has_key(self.name):
            raise ValueError("Something called " + self.name + " already "
                             "exists. Please choose a different name.")

        model.objs[self.name] = self

        for obj in self.objects:
            obj.name = self.name + '.' +  obj.name
            model.add(obj)

    def build(self, model, dt):
        pass
