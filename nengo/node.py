from output import Output

def is_node(obj):
    return isinstance(obj, (Node, TimeNode))

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
            if not callable(output):
                outval = output
                def output():
                    return outval
                outfunc = output
            else:
                outfunc = output
            self.add_output(outfunc)
        
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
        self.outputs[Output(dimensions=len(func()), name=self.name + ":" + func.__name__)] = func
        
    def get(self, name):
        found = [self for x in self.inputs if x == name] + [x for x in self.outputs if x.name == name]
                
        if len(found) > 1:
            print "Warning, found more than one input or output with same name"
        
        return found[0]

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
            new_state[output] = self.outputs[output]()

class TimeNode(Node):
    """
    A custom-node for feeding a function-of-time into the network.
    """
    def __init__(self, name, output):
        Node.__init__(self, name)
        self.outputs = {Output(dimensions=len(output(0.0)), name=self.name + ":" + output.__name__):output}
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
            state_tm1[output] = self.outputs[output](self.t)
            
class FileNode(Node):
    def __init__(self, name, output):
        Node.__init__(self, name)
        
class DictNode(TimeNode):
    def __init__(self, name, output):
        self.data = output
        TimeNode.__init__(self, name, self.output)
        
    def output(self, time):
        output_time = 0.0
        for t in self.data:
            if t < time and time-t < time-self.data[output_time]:
                output_time = t
            
        return self.data[output_time]

