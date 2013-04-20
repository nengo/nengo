"""This is a file to create and test the basal ganglia template.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from .. import nef_theano as nef
from .. import templates

net = nef.Network('BG Test')
def func(x):
    return [math.sin(x), .5,.2]
net.make_input('in', value=func)
templates.basalganglia.make_basal_ganglia(
    net=net, name='BG', dimensions=3)

net.connect('in', 'BG.input')

timesteps = 1000
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
BGp = net.make_probe('BG.output', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ioff(); plt.close(); 
plt.subplot(2,1,1)
plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
plt.subplot(2,1,2)
plt.plot(BGp.get_data()); plt.title('BG.output')
plt.tight_layout()
plt.show()
