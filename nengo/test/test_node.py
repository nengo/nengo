import nengo

class TestNode(nengo.Node):
    def step(self):
        print "test"
        
m = nengo.Model("test")
t = TestNode()
m.add(t)
m.run(0.1)

