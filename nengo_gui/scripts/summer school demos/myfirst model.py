import nengo

model = nengo.Network()
with model:
    a = nengo.Ensemble(100, dimensions=1, label="integrator")
    input = nengo.Node(0, label="input")

    nengo.Connection(input, a)
    
    nengo.Probe(a)
    nengo.Probe(a, 'spikes')


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 1.9028588839523086
gui[model].offset = 290.15293973677046,346.692822320954
gui[a].pos = -22.284, -115.488
gui[a].scale = 1.000
gui[input].pos = -127.177, -111.937
gui[input].scale = 1.000
