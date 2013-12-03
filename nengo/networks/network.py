import nengo

class Network(object):
    def __init__(self, *args, **kwargs):
        self.label = kwargs.pop("label", "Network")
        self.objects = []
        self.make(*args, **kwargs)
        
        #add self to current context
        nengo.context.add_to_current(self)

    def add(self, obj):
        self.objects.append(obj)
        return obj

    def make(self, *args, **kwargs):
        raise NotImplementedError("Networks should implement this function.")

    def add_to_model(self, model):
        for obj in self.objects:
            if not isinstance(obj, (nengo.Connection, nengo.ConnectionList)):
                obj.label = self.label + '.' +  obj.label
            model.add(obj)
            
    def __enter__(self):
        nengo.context.append(self)
        
    def __exit__(self, exception_type, exception_value, traceback):
        nengo.context.pop()
