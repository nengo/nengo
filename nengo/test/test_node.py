import nengo
import math

class TestNode(nengo.Node):
    def init(self):
        self.add_input("x")
        self.add_input("y")
    
    def step(self):
        print self.x, self.y
        
m = nengo.Model("test")

input1 = m.make_node("input1", lambda t : [math.sin(t)])
input2 = m.make_node("input2", {0:[0.1], 0.05:[0.2], 0.08:[0.3]})

test = TestNode("testnode")
m.add(test)

m.connect("input1:<lambda>", "testnode:x")
m.connect(input2, "testnode:y")

m.run(0.1)

