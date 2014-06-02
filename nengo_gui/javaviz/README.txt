First rough version of Java visualizer for new Nengo

Installation:
- grab a latest copy of nengo 1.4
- copy the stuff from nengo-java/ into nengo 1.4's "python" directory
- run "nengo-cl python/rpyc/scripts/rpyc_classic.py"

Now we are all set to go.

To use the visualizer, do something like this:

-------------
model = nengo.Model('Visualization Test')
with model:
    input = nengo.Node(np.sin, label='input')
    
    a = nengo.Ensemble(neurons=100, dimensions=1, label='A')
    b = nengo.Ensemble(neurons=100, dimensions=1, label='B')
    nengo.Connection(input, a, filter=0.01)
    nengo.Connection(a, b, filter=0.01, function=lambda x: [x[0]*x[0]])
   
import javaviz   
javaviz.View(model)   # this is all you need to do!

sim = nengo.Simulator(model)
sim.run(100000)
-------------------    

