from .. import objects
from .. import context

class Network(object, context.Context):
    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop("label", "Network")
        self.objects = []
        self.make(*args, **kwargs)
        
        #add self to current context
        context.add_to_current(self)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            obj.label = self.label + '.' +  obj.label
            model.add(obj)
