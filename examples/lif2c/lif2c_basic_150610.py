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
    n_lif2c_dim = 2
    n_gain_pop = 80   
    
    encoders = nengo.dists.UniformHypersphere(surface=True).sample(n_lif2c_pop, n_lif2c_dim)    
    
    lif_model = nengo.LIF2C()
    gain, bias = lif_model.gain_bias(intercepts=nengo.dists.Uniform(0.2, 0.9).sample(n_lif2c_pop),
                                     max_rates=nengo.dists.Uniform(50, 80).sample(n_lif2c_pop))
                                     
    
    lif2c = nengo.Ensemble(neuron_type=nengo.LIF2C(g_c=5.0, g_l=0.5, C_A=0.05, C_S=0.2, V_T=0.2, V_R=-0.2, bias=bias), 
                           n_neurons=n_lif2c_pop, dimensions=n_lif2c_dim, radius=1, encoders=encoders, bias=bias, gain=gain)
    gain_pop = nengo.Ensemble(n_neurons=n_gain_pop, dimensions=n_lif2c_dim)
    
    def input_steps(t):
        if t < 3:
            return -1 * np.ones((n_lif2c_dim,))
        elif t < 6:
            return 0 * np.ones((n_lif2c_dim,))
        else:
            return 1 * np.ones((n_lif2c_dim,))
            
    def gain_steps(t):
        if mod(t, 3) < 1:
            return 0* np.ones((n_lif2c_dim,))
        elif mod(t, 3) < 2:
            return 0 * np.ones((n_lif2c_dim,))
        else:
            return 1 * np.ones((n_lif2c_dim,))
    #input_node = nengo.Node(lambda t: np.ones((n_lif2c_dim,)) * np.sin(5.0 * t))
    #gain_node = nengo.Node(lambda t: np.ones((2,)) * sin(2.0*t)) #(t > 1.0))
    input_node = nengo.Node(input_steps)
    gain_node = nengo.Node(gain_steps) #(t > 1.0))
    
    probe_g_shunt = nengo.Probe(lif2c.neurons, 'g_shunt')        
    probe_spikes = nengo.Probe(lif2c.neurons)
    probe_gain_spikes = nengo.Probe(gain_pop.neurons)
    probe_V_A = nengo.Probe(lif2c.neurons, 'V_A')
    probe_V_S = nengo.Probe(lif2c.neurons, 'V_S')
    probe_lif2c = nengo.Probe(lif2c, synapse=0.1)
    probe_gain = nengo.Probe(gain_pop, synapse=0.1)
    
    nengo.Connection(input_node, lif2c)
    #nengo.Connection(input_node, lif2c.neurons, transform=0.015*np.ones((n_lif2c_pop,1)))

    nengo.Connection(gain_node, gain_pop)
    
    #g_shunt_weights = .12 * abs(encoders[:,:1]) #np.ones((n_lif2c_pop, 1))
    g_shunt_weights = .2 * encoders[:, :1]
    #g_shunt_weights[g_shunt_weights < 0] = 0
    nengo.Connection(gain_pop[0], lif2c.neurons, synapse=0.01, transform=g_shunt_weights, target='g_shunt')

#    
sim = nengo.Simulator(m)
sim.run(10.0)

#%%
figure(1)
clf()
plot(sim.trange(), sim.data[probe_g_shunt][:, :])

# below 0
g_shunt_below0 = zeros(n_lif2c_pop)

g_shunt_below0 = sim.data[probe_g_shunt][2500, :] < 0
    
figure('below0 v')
#plot(sim.trange(), sim.data[probe_V_S][:, g_shunt_below0[:5]])    

 #%%   

figure(2)
clf()
plot(sim.trange(), sim.data[probe_spikes][:, :1])

figure(3)
clf()
plot(sim.trange(), sim.data[probe_gain_spikes][:, :1])

figure(4)
clf()
plot(sim.trange(), sim.data[probe_V_S][:, :5])

figure(5)
clf()
plot(sim.trange(), sim.data[probe_V_A][:, :5])

figure(6)
clf()
plot(sim.trange(), sim.data[probe_gain][:, :1])

figure(7)
clf()
#plot(sim.trange(), -sin(2.0 * sim.trange()) * sin(5.0 * sim.trange()))
plot(sim.trange(), [input_steps(t) * np.ceil(-(gain_steps(t)-1)) for t in sim.trange()])
plot(sim.trange(), 1.0 * sim.data[probe_lif2c][:, :1])

figure(8)
plot(sim.trange(), 1.0 * sim.data[probe_lif2c][:, 1:2])
#if __name__=='__main__':
#    import nengo_gui
#    nengo_gui.Viz(__file__).start()
# 
    