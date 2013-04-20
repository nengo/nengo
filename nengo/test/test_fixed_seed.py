"""This is a file to test the fixed_seed parameter, which should make 
identical ensembles.

"""

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

net = nef.Network('Array Test', fixed_seed=5)
net.make_input('in', [1], zero_after_time=1.0)
net.make('A', 50, 1)
net.make('B', 50, 1)
net.make_array('AB', 50, 2)


net.connect('in', 'A')
net.connect('in', 'B')
net.connect('in', 'AB', index_post=[0, 1])

timesteps = 200
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
ABp = net.make_probe('AB', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ioff(); plt.close(); 
plt.subplot(5,1,1)
plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
plt.subplot(5,1,2)
plt.plot(Ap.get_data()); plt.title('A')
plt.subplot(5,1,3)
plt.plot(Bp.get_data()); plt.title('B')
plt.subplot(5,1,4)
plt.plot(ABp.get_data()[:,0]); plt.title('AB[0]')
plt.subplot(5,1,5)
plt.plot(ABp.get_data()[:,1]); plt.title('AB[1]')
plt.show()
