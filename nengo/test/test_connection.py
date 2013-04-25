import nengo

import math
import numpy as np

m = nengo.Model("test_connection")

input = m.make_node("input", np.array([.5])[:,None])# lambda x: [math.sin(x)])

pop = m.make_ensemble("pop", 100, 1)
m.make_ensemble('b', 10, 1)

m.connect("input:output", pop)
m.connect('pop','b')

m.probe('input:output')
m.probe('pop:output')

m.build()
probes = m.run(6, dt=.001)
        
import matplotlib.pyplot as plt
plt.close(); plt.ion(); plt.hold(1)
plt.plot(np.array(probes[1].data)[:,0,0])
plt.plot(np.array(probes[2].data)[:,0,0])
plt.plot(np.sin(np.arange(0,6,.001)))
