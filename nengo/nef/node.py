import numpy
import math


class Filter(object):
    def __init__(self, tau):
        self.tau=tau
        self.state=None
    def update(self, value, dt):
        if self.state is None:
            self.state=value
        else:
            decay=math.exp(-dt/self.tau)
            self.state=decay*self.state+(1-decay)*value
        return self.state    


class Input(object):
    def __init__(self, dimensions, filter=None):
        self.set([0]*dimensions)
        self.filter=filter
    def update(self, dt):
        if self.filter is not None: 
            self.state=self.filter.update(self.raw, dt)
        else:
            self.state=self.raw    
    def set(self, value):
        self.raw=numpy.array(value)
    def get(self):
        return self.state
    def reset(self):
        self.set([0]*len(self.state))
            
        
class Output(object):
    def __init__(self, dimensions):
        self.state=[0]*dimensions
    def set(self, value):
        self.state=value
    def get(self):
        return self.state
    def reset(self):
        self.state=[0]*len(self.dimensions)
        
    


class Node(object):
    def __init__(self, name=None):
        self.t=None
        self.name=name
        self._inputs=[]
        self._outputs=[]
    
    def make_input(self, dimensions, filter=None):
        input=Input(dimensions, filter)
        self._inputs.append(input)
        return input
    
    def make_output(self, dimensions):
        output=Output(dimensions)
        self._outputs.append(output)
        return output
    
    def tick(self):
        pass
        
    def start(self):
        pass    
        
    def reset(self):
        for input in self._inputs:
            input.reset()
        for output in self._outputs:    
            output.reset()
        self.t=None    
       
    def update(self, dt):
        if self.t is None:
            self.start()
            self.t=0
        for input in self._inputs:
            input.update(dt)
        self.tick()
        self.t+=dt
        












if __name__=='__main__':
    class SquaringNode(Node):
        def __init__(self, dimensions):
            Node.__init__(self)
            self.input = self.make_input(dimensions=dimensions, filter=Filter(tau=0.01))
            self.output = self.make_output(dimensions=dimensions)
        def tick(self):          
            self.output.set(self.input.get()**2)
        


    n=SquaringNode(dimensions=3)
    
    for i in range(100):
        n.input.set([i*0.01, i, [0, 1][i<50]])
        n.update(0.001)
        print i, n.output.get()
   
   
    class DelayNode(Node):
          def __init__(self, steps, dimensions):
              Node.__init__(self)
              self.steps = steps
              self.dimensions = dimensions
              self.input = self.make_input(dimensions=dimensions)
              self.output = self.make_output(dimensions=dimensions)
          def start(self):
              self.delay = [[0]*self.dimensions for i in range(self.steps)]
          def tick(self):
              self.output.set(self.delay[0])
              self.delay = self.delay[1:] + [self.input.get()]         
       

    n=DelayNode(steps=1, dimensions=1)
    
    for i in range(100):
        n.input.set([i])
        n.update(0.001)
        print i, n.output.get()


    class InputNode(Node):
        def __init__(self, function, name=None):
            Node.__init__(self, name)
            self.function = function
            self.output = self.make_output(dimensions=len(function(0)))
        def tick(self):
            self.output.set(self.function(self.t))    
                        
                        
    class StaticNode(Node):
        def __init__(self, value, name=None):
            Node.__init__(self, name)
            self.output = self.make_output(dimensions=len(value))
            self.output.set(value)
            
            
    class PiecewiseNode(Node):
        def __init__(self, values, name=None):
            Node.__init__(self, name)
            
            self.data=list(values.items())
            self.data.sort()
            self.index=-1
            
            self.output = self.make_output(dimensions=self.data[0][0])
            
        def tick(self):
            while self.index+1<len(self.data) and self.t>=self.data[self.index+1][0]:
                     
                self.index += 1
                self.output.set(self.data[self.index][1])    
                            
            
            
                            



    def wave(t):
        return [math.sin(t)]
    input=InputNode(wave, name='input')
    for i in range(100):
        input.update(0.1)
        print i, input.output.get()
    input=StaticNode([2, 3, 4], name='input')
    for i in range(100):
        input.update(0.1)
        print i, input.output.get()


    input=PiecewiseNode({0:[8, 3], 0.3:[4, 1], 0.37321: [1, 1]}, name='input')
    for i in range(100):
        input.update(0.01)
        print i, input.t, input.output.get()

