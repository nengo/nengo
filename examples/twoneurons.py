import nengo
from nengo.objects import Uniform
import numpy as np

model = nengo.Model('Two Neurons')

n = nengo.Ensemble(nengo.LIF(2), #
                    dimensions=1, # Representing a scalar
                    intercepts=Uniform(-.5, -.5), #Set the intercepts at .5
                    max_rates=Uniform(100,100), #Set the max firing rate at 100hz
                    encoders=[[1],[-1]]) #One 'on' and one 'off' neuron



sin = nengo.Node( output=lambda t: np.sin(8*t))

nengo.Connection(sin, n, filter = 0.01)

sin_p = nengo.Probe(sin, 'output') # The original input
n_spikes = nengo.Probe(n, 'spikes') # Raw spikes from each neuron
#nengo.Probe(n, 'voltages') # Subthreshold soma voltages of the neurons
n_output = nengo.Probe(n, 'decoded_output', filter = 0.01) # Spikes filtered by a 10ms post-synaptic filter

sim = nengo.Simulator(model) #Create a simulator
sim.run(1) # Run it for 5 seconds


import matplotlib.pyplot as plt
# Plot the decoded output of the ensemble
t = sim.data(model.t_probe) #Get the time steps
plt.plot(t, sim.data(n_output))
plt.plot(t, sim.data(sin_p))
plt.xlim(0,1)

# Plot the spiking output of the ensemble
from nengo.matplotlib import rasterplot
plt.figure(figsize=(10,8))
plt.subplot(221)
rasterplot(t, sim.data(n_spikes), colors=[(1,0,0), (0,0,0)])
plt.xlim(0,1)
plt.yticks((1, 2), ("On neuron", "Off neuron"))

# Plot the soma voltages of the neurons
#plt.subplot(222)
#data = sim.data('Neurons.voltages')
#plt.plot(t, data[:,0]+1, 'r')
#plt.plot(t, data[:,1], 'k')
#plt.yticks(())
#plt.axis([0,1,0,2])
#plt.subplots_adjust(wspace=0.05)

plt.show()