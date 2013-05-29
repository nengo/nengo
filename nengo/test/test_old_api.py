import numpy as np
from nengo.old_api import Network
from matplotlib import pyplot as plt

def test_nengo2model1():

    net = Network('Runtime Test', dt=0.001, seed=123)
    print 'make_input'
    net.make_input('in', value=np.sin)
    print 'make A'
    net.make('A', 1000, 1)
    print 'connecting in -> A'
    net.connect('in', 'A')
    net_A_probe = net.make_probe('A', dt_sample=0.01, pstc=0.01)

    net._make_simulator()
    for i in range(10):
        net.sim.step()
        print net.sim.signals

    net.run(1.0)
    net_data = np.asarray(net_A_probe.get_data()).flatten()

    #plt.plot(net_data)
    #plt.show()

    print net_data

