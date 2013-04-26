import nengo
import math

class TestNode(nengo.Node):
    def init(self):
        self.add_input("x")
        self.add_input("y")
        self.add_input("z")
    
    def step(self):
        print self.x, self.y, self.z
        
f = open("test.csv", "w")
f.write("0.0,0.1,0.1\n")
f.write("0.05,0.0,0.2\n")
f.close()        

m = nengo.Model("test")

input1 = m.make_node("input1", lambda t : [math.sin(t)])
input2 = m.make_node("input2", {0.05:[0.2], 0.08:[0.3]})
input3 = m.make_node("input3", "test.csv")

test = TestNode("testnode")
m.add(test)

m.connect("input1:output", "testnode:x")
m.connect(input2, "testnode:y")
m.connect(input3, "testnode:z")

m.run(0.1)

