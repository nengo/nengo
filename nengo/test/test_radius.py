"""This is a test file to test the radius parameter of ensembles.

Need to test the radius both on identity, linear, and non-linear 
projections. It affects 3 places: the termination (scales input),
the origin (scales output), and when computing decoders (scales 
the function being computed so that it has the proper shape inside
unit length).

"""
import math

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

def sin3(x):
    return math.sin(x) * 3

def pow(x):
    return [xval**2 for xval in x]

def mult(x):
    return [xval*2 for xval in x]

net = nef.Network('Encoder Test')
net.make_input('in', value=sin3)
net.make('A', 1000, 1, radius=5)
net.make('B', 300, 1, radius=.5)
net.make('C', 1000, 1, radius=10)
net.make('D', 300, 1, radius=6)

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('A', 'C', func=pow)
net.connect('A', 'D', func=mult)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
Dp = net.make_probe('D', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps * dt_step)

plt.ioff(); plt.clf(); plt.hold(1);
plt.plot(Ip.get_data())
plt.plot(Ap.get_data())
plt.plot(Bp.get_data())
plt.plot(Cp.get_data())
plt.plot(Dp.get_data())
plt.legend(['Input','A','B','C','D'])
plt.tight_layout()
plt.show()
