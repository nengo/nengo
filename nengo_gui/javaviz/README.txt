First rough version of Java visualizer for new Nengo

Installation:
- grab a latest copy of nengo 1.4
- copy the stuff from nengo-java/ into nengo 1.4's "python" directory
- run "nengo-cl python/rpyc/scripts/rpyc_classic.py"

You will also need rpyc installed in your normal Python installation.  To do
this, run "pip install rpyc"

Now we are all set to go.

To use the visualizer, do something like this:

-------------
model = nengo.Network('Visualization Test')
with model:
    input = nengo.Node(np.sin, label='input')
    
    a = nengo.Ensemble(100, dimensions=1, label='A')
    b = nengo.Ensemble(100, dimensions=1, label='B')
    nengo.Probe(b)           # to indicate this data should be visualized
    nengo.Probe(b, 'spikes') # to indicate this data should be visualized
    nengo.Connection(input, a, synapse=0.01)
    nengo.Connection(a, b, synapse=0.01, function=lambda x: [x[0]*x[0]])
   
import javaviz   
javaviz.View(model)   # this is all you need to do!

sim = nengo.Simulator(model)
sim.run(1000)
-------------------    

