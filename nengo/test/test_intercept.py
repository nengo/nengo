"""This is a file to test the intercept parameter on ensembles"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

build_time_start = time.time()

inter = .5; dx = .001

net = nef.Network('Intercept Test')
net.make_input('in', math.sin)
net.make('A', 100, 1)
net.make('B', 100, 1, intercept=(inter, 1.0)) 
#eval_points=np.array(
#    [np.arange(-1,-inter,dx), np.arange(inter,1,dx)]).flatten())

net.connect('in', 'A')
net.connect('A', 'B')

timesteps = 1000
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)

build_time_end = time.time()

print "starting simulation"
net.run(timesteps * dt_step)

sim_time_end = time.time()
print "\nBuild time: %0.10fs" % (build_time_end - build_time_start)
print "Sim time: %0.10fs" % (sim_time_end - build_time_end)

plt.ioff(); plt.close(); plt.hold(1)
plt.plot(t, Ip.get_data())
plt.plot(t, Ap.get_data())
plt.plot(t, Bp.get_data())
plt.legend(['Input', 'A', 'B'])
plt.tight_layout()
plt.show()

