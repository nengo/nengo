import nengo
import math

class TestNode(nengo.Node):
    def init(self):
        self.add_input("x")
    
    def step(self):
        print self.x
        
m = nengo.Model("test")

def output(t):
    print t
    return [math.sin(t)]

input = m.make_node("input", output)

t = TestNode("testnode")
m.add(t)

m.connect("input:output", "testnode:x")

m.run(0.1)

