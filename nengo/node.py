from output import Output

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
        self.connections = [] #list of input connections
        self.inputs = [] #list of variables in this node (for net.connect to connect to)
        
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
        self.outputs[Output(dimensions=len(func()), 
            name=self.name + ":" + func.__name__)] = func
        
    def get(self, name):
        search = [self for x in self.inputs if x == name] + [x for x in self.outputs if x.name == name]
                
        if len(search) > 1:
            print "Warning, found more than one object with same name"
        if len(search) == 0:
            print name + " not found in node.get"
            return None
        return search[0]

    def _build(self, state, dt):
        for output in self.outputs:
            state[output] = self.outputs[output]()

    def _reset(self, state):
        for output in self.outputs:
            state[output] = self.outputs[output]()
        
        self.reset()

    def _step(self, old_state, new_state, dt):
        for c in self.connections:
            setattr(self, c.post.split(":")[-1], c.get_post_input(old_state, dt))
        
        self.step()
        
        for output in self.outputs:
            new_state[output] = \
                np.asarray(self.outputs[output]())[:,None]

class TimeNode(Node):
    """
    A custom-node for feeding a function-of-time into the network.
    """
    def __init__(self, name, output):
        Node.__init__(self, name)
        self.outputs = {Output(dimensions=len(output(0.0)), 
            name=self.name + ":" + output.__name__):output}
        self.t = 0.0

    def _build(self, state, dt):
        self._reset(state)

    def _reset(self, state):
        self.t = 0
        for output in self.outputs:
            state[output] = self.outputs[output](self.t)

    def _step(self, state_t, state_tm1, dt):
        self.t += dt
        
        for c in self.connections:
            self.inputs[c.post] = c.get_post_input()
        
        self.step()    
        
        for output in self.outputs:
            state_tm1[output] = \
                np.asarray(self.outputs[output](self.t))[:,None]
            
class ValueNode(Node):
    def __init__(self, name, output):
        self.val = output
        Node.__init__(self, name, self.output)
        
    def output(self):
        return self.val
        
class DictNode(TimeNode):
    def __init__(self, name, output):
        self.data = output
        TimeNode.__init__(self, name, self.output)
        
    def output(self, time):
        output_time = -1
        for t in self.data:
            if t <= time and time-t < time-output_time:
                output_time = t
            
        if output_time == -1:
            return [0.0 for _ in range(len(self.data[self.data.keys()[0]]))]
        else:
            return self.data[output_time]
    
class FileNode(DictNode):
    def __init__(self, name, output):
        
        data = {}
        f = open(output, "r")
        for line in f:
            split = line.split(",")
            data[float(split[0])] = [float(x) for x in split[1:]]
        f.close() 
        
        DictNode.__init__(self, name, data)
        
        self.val = np.asarray(output)
        Node.__init__(self, name, self.output)
        
    def output(self):
        return self.val
