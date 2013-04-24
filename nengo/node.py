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
        self.inputs = {} #map input names to internal variables
        self.connections = {} #map connections to internal variables
        
        if output != None:
            if not callable(output):
                outval = output
                def output():
                    return outval
                outfunc = output
            else:
                outfunc = output
            self.add_output(outfunc)
        
    def step(self):
        """This is the function subclasses should override to insert arbitrary
        code into Node execution."""
        pass
    
    def reset(self):
        """Override this to define reset behaviour for inserted code."""
        pass
    
    def add_input(self, name=None):
        if name == None:
            name = "input"
        setattr(self, name, None)
        inputname = self.name + ":" + name
        self.inputs[inputname] = getattr(self,name)
        
    def add_connection(self, c):
        if c.post == self:
            #no input specified, so make a new input for this connection
            self.add_input()
            c.post = "input"
            
        self.connections[c] = self.inputs[c.post]
        
    def add_output(self, func):
        self.outputs[Output(dimensions=len(func()), name=func.__name__)] = func
        
    def get(self, name):
        """This method should return the object(s) corresponding to name."""
        
        found = []
        for x in self.inputs:
            if x == name:
                found += [self] #inputs are always just referenced via the parent object
        for x in self.outputs:
            if x.name == name:
                found += [x] #return the Output() object
        return found
        
    def _build(self, state, dt):
        for output in self.outputs:
            state[output] = self.outputs[output]()

    def _reset(self, state_t):
        for output in self.outputs:
            state_t[output] = self.outputs[output]()
        
        self.reset()

    def _step(self, state_t, state_tm1, dt):
        for c in self.connections:
            self.inputs[c.post] = c.get_post_input()
        
        self.step()
        
        for output in self.outputs:
            state_tm1[output] = self.outputs[output]()

class TimeNode(Node):
    """
    A custom-node for feeding a function-of-time into the network.
    """
    def __init__(self, name, func):
        self.name = name
        self._outputs = {Output(dimensions=len(func(0.0))):func}
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

