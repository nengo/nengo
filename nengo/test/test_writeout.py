"""This is a file to test the net.write_to_file method
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from neo import hdf5io

from .. import nef_theano as nef

build_time_start = time.time()

net = nef.Network('Write Out Test')
net.make_input('in', math.sin)
net.make('A', 50, 1)
net.make('B', 5, 1)

net.connect('in', 'A')
net.connect('in', 'B')

timesteps = 100
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step)
BpSpikes = net.make_probe('B', data_type='spikes', dt_sample=dt_step)

build_time_end = time.time()

print "starting simulation"
net.run(timesteps*dt_step)

sim_time_end = time.time()
print "\nBuild time: %0.10fs" % (build_time_end - build_time_start)
print "Sim time: %0.10fs" % (sim_time_end - build_time_end)

net.write_data_to_hdf5()

# open up hdf5 file 
iom = hdf5io.NeoHdf5IO(filename='data.hd5')

print iom.get_info()
# wtf i know right?
block_as = iom.read_analogsignal()
segment_as = block_as.segments[0]
block_st = iom.read_spiketrain()
segment_st = block_st.segments[0]

import matplotlib.pyplot as plt
plt.clf();
plt.subplot(211); plt.title('analog signal'); plt.hold(1)
legend = []
for asig in segment_as.analogsignals:
    plt.plot(asig)
    legend.append(asig.annotations['target_name'])
plt.legend(legend)
plt.subplot(212); plt.title('spike train')
legend = []
for i, ssig in enumerate(segment_st.spiketrains):
    if len(ssig) == 0: continue
    plt.vlines(ssig, 1, 0)
    legend.append('neuron %d'%i)
plt.legend(legend)
plt.tight_layout()
plt.show()

# close up hdf5 file
iom.close()

