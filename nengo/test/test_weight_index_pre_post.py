"""This is a test file to test the weight, index_pre, and index_post parameters
on the connect function. 
"""

import math

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

net = nef.Network('Weight, Index_Pre, and Index_Post Test')
net.make_input('in', value=math.sin)
net.make('A', 300, 1)
net.make('B', 300, 1)
net.make('C', 400, 2)
net.make('D', 800, 3)
net.make('E', 400, 2)
net.make('F', 400, 2)

net.connect('in', 'A', weight=.5)
net.connect('A', 'B', weight=2)
net.connect('A', 'C', index_post=1)
net.connect('A', 'D')
net.connect('C', 'E', index_pre=1)
net.connect('C', 'F', index_pre=1, index_post=0)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
Dp = net.make_probe('D', dt_sample=dt_step, pstc=pstc)
Ep = net.make_probe('E', dt_sample=dt_step, pstc=pstc)
Fp = net.make_probe('F', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

plt.ioff(); plt.close(); 
plt.subplot(711); plt.title('Input')
plt.plot(Ip.get_data())
plt.subplot(712); plt.title('A = Input * .5')
plt.plot(Ap.get_data())
plt.subplot(713); plt.title('B = A * 2')
plt.plot(Bp.get_data())
plt.subplot(714); plt.title('C(0) = 0, C(1) = A')
plt.plot(Cp.get_data())
plt.subplot(715); plt.title('D(0:2) = A')
plt.plot(Dp.get_data())
plt.subplot(716); plt.title('E(0:1) = C(1)')
plt.plot(Ep.get_data())
plt.subplot(717); plt.title('F(0) = C(1)')
plt.plot(Fp.get_data())
plt.tight_layout()
plt.show()
