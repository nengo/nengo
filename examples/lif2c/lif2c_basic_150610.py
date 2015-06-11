# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:38:28 2015

@author: Paxon
"""

import nengo
import pylab

with nengo.Network() as m:
    lif2c = nengo.Ensemble(neuron_type=nengo.LIF2C(), n_neurons=1, dimensions=1, radius=2)
    iz = nengo.Ensemble(neuron_type=nengo.Izhikevich(), n_neurons=40, dimensions=1)
    alif = nengo.Ensemble(neuron_type=nengo.AdaptiveLIF(), n_neurons=40, dimensions=1)
    
    input_node = nengo.Node([0])
    gain_node = nengo.Node([0])
    
    nengo.Connection(input_node, lif2c)
    
    probe_g_shunt = nengo.Probe(lif2c.neurons, 'g_shunt')        
    probe_spikes = nengo.Probe(lif2c.neurons)
    probe_V_A = nengo.Probe(lif2c.neurons, 'V_A')
    probe_V_S = nengo.Probe(lif2c.neurons, 'V_S')
    
    
    g_shunt_weights = ones(1)
    nengo.Connection(gain_node, lif2c.neurons, synapse=0.1, transform=g_shunt_weights)

    
sim = nengo.Simulator(m)
sim.run(1.0)

figure(1)
clf()
plot(sim.trange(), sim.data[probe_g_shunt][:, :5])


if __name__=='__main__':
    import nengo_gui
    nengo_gui.Viz(__file__).start()
    
    