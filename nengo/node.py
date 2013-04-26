from output import Output
import inspect
import numpy as np

def is_node(obj):
    return isinstance(obj, Node)

class Node(object):
    """
    A custom-node for feeding a function into the network.
    """

    def __init__(self, name, output=None):
        self.name = name

        self.outputs = {} #map output objects to output functions
        self.time_outputs = {} #map output objects to output functions of time
        self.connections = [] #list of input connections
        self.inputs = [] #list of variables in this node (for net.connect to connect to)
        self.t = 0.0
        
        if output != None:
            self.add_output(output)
        
        self.init()
        
    def step(self):
        """This is the function subclasses should override to insert arbitrary
        code into Node execution."""
        pass
    
    def reset(self):
        """Override this to define reset behaviour for inserted code."""
        pass
    
    def init(self):
        """Override this to define initialization behaviour."""
        pass
    
    def add_input(self, name):
        self.inputs += [self.name + ":" + name]
        
    def add_connection(self, c):
        if c.post == self:
            #no input specified, so make a new input for this connection
            c.post = self.name + ":input"
            self.add_input("input")
            
        self.connections += [c]
        
    def add_output(self, func):
        
        if func.__name__ == "<lambda>":
            funcname = "output"
        else:
            funcname = func.__name__
        
        func_args = inspect.getargspec(func).args
            
        if len(func_args) == 1:
            o = Output(dimensions=len(func(0.0)), name=self.name + ":" + funcname)
            self.time_outputs[o] = func
        elif len(func_args) == 0:
            o = Output(dimensions=len(func()), name=self.name + ":" + funcname)
            self.outputs[o] = func
        else:
            print "Nodes only accepts output functions with 0 or 1 arguments"
        
    def get(self, name):
        search = [self for x in self.inputs if x == name] + \
                [x for x in self.outputs.keys()+self.time_outputs.keys() if x.name == name]
                
        if len(search) > 1:
            print "Warning, found more than one object with same name"
        if len(search) == 0:
            print name + " not found in node.get"
            return None
        return search[0]

    def _build(self, state, dt):
        self._reset(state)

    def _reset(self, state):
        for o in self.outputs:
            state[o] = np.asarray(self.outputs[o]())
        
        for o in self.time_outputs:
            state[o] = np.asarray(self.time_outputs[o](self.t))
            
        self.t = 0
        
        self.reset()

    def _step(self, old_state, new_state, dt):
        self.t += dt
        
        for c in self.connections:
            setattr(self, c.post.split(":")[-1], c.get_post_input(old_state, dt))
        
        self.step()
        
        for output in self.outputs:
            new_state[output] = np.asarray(self.outputs[output]())
                
        for output in self.time_outputs:
            new_state[output] = np.asarray(self.time_outputs[output](self.t))
            
class ValueNode(Node):
    def __init__(self, name, output_val):
        self.val = output_val
        
        def output():
            return self.val
        
        Node.__init__(self, name, output)
        
class DictNode(Node):
    def __init__(self, name, output_dict):
        self.data = output_dict
        
        def output(time):
            output_time = -1
            for t in self.data:
                if t <= time and time-t < time-output_time:
                    output_time = t
                
            if output_time == -1:
                return [0.0 for _ in range(len(self.data[self.data.keys()[0]]))]
            else:
                return self.data[output_time]
        
        Node.__init__(self, name, output)
        
    
class FileNode(DictNode):
    def __init__(self, name, output_filename):
        
        data = {}
        f = open(output_filename, "r")
        for line in f:
            split = line.split(",")
            data[float(split[0])] = [float(x) for x in split[1:]]
        f.close() 
        
        DictNode.__init__(self, name, data)
