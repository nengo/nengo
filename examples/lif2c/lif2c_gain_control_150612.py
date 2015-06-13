# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:38:28 2015

@author: Paxon
"""

import nengo
import numpy as np
import pylab

with nengo.Network() as m:

    n_lif2c_pop = 80
    n_lif2c_dim = 8
    n_gain_pop = 80   
    
    encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_lif2c_pop, n_lif2c_dim)    
    
    lif2c = nengo.Ensemble(neuron_type=nengo.LIF2C(g_c=5.0, g_l=0.5, C_A=0.05, C_S=0.2, V_T=0.2, V_R=-0.2), 
                           n_neurons=n_lif2c_pop, dimensions=n_lif2c_dim, radius=2, encoders=encoders)
    
    def input_steps(t):
        if t < 2:
            return -2 * np.ones((n_lif2c_dim,))
        elif t < 4:
            return 1 * np.ones((n_lif2c_dim,))
        elif t < 6:
            return 2 * np.ones((n_lif2c_dim,))
        else:
            return 4 * np.ones((n_lif2c_dim,))
    #input_node = nengo.Node(lambda t: np.ones((n_lif2c_dim,)) * np.sin(5.0 * t))
    #gain_node = nengo.Node(lambda t: np.ones((2,)) * sin(2.0*t)) #(t > 1.0))
    input_node = nengo.Node(input_steps)
    
    probe_g_shunt = nengo.Probe(lif2c.neurons, 'g_shunt')        
    probe_spikes = nengo.Probe(lif2c.neurons)
    probe_V_A = nengo.Probe(lif2c.neurons, 'V_A')
    probe_V_S = nengo.Probe(lif2c.neurons, 'V_S')
    probe_lif2c = nengo.Probe(lif2c, synapse=0.1)
    
    nengo.Connection(input_node, lif2c)
    #nengo.Connection(input_node, lif2c.neurons, transform=0.015*np.ones((n_lif2c_pop,1)))
    
    #g_shunt_weights = .12 * abs(encoders[:,:1]) #np.ones((n_lif2c_pop, 1))
    #g_shunt_weights = .1 * encoders[:, :1]
    #g_shunt_weights[g_shunt_weights < 0] = 0
    #nengo.Connection(gain_pop[0], lif2c.neurons, synapse=0.01, transform=g_shunt_weights, target='g_shunt')
    
    gain_control_weights = 5e-7 * ones((n_lif2c_pop, n_lif2c_pop))
    nengo.Connection(lif2c.neurons, lif2c.neurons, synapse=0.01, 
                     transform=gain_control_weights, target='g_shunt')
    

#    
sim = nengo.Simulator(m)
sim.run(10.0)

#%%
figure(1)
clf()
plot(sim.trange(), sim.data[probe_g_shunt][:, :1])
title('g_shunt')

figure(2)
clf()
plot(sim.trange(), sim.data[probe_spikes][:, :1])
title('spikes')

figure(4)
clf()
plot(sim.trange(), sim.data[probe_V_S][:, :5])
title('V_S')

figure(5)
clf()
plot(sim.trange(), sim.data[probe_V_A][:, :5])
title('V_A')


figure(7)
clf()
plot(sim.trange(), 1.0 * sim.data[probe_lif2c][:, :1])
title('dimension')

figure(8)
plot(sim.trange(), 1.0 * sim.data[probe_lif2c][:, 1:2])
title('dimension')

#if __name__=='__main__':
#    import nengo_gui
#    nengo_gui.Viz(__file__).start()
# 
    