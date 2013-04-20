"""This test file is for checking the eval_points parameter added to the
ensemble and origin constructors.

An ensemble can be created with a set of default eval_points for
every origin to use, or an origin can be called with
a specific set of eval_points to use for optimization. 
   
This tests:

1. creating origin w/ eval_points
2. creating ensemble w/ eval_points
3. creating ensemble w/ eval_points, creating origin w/ eval_points
4. creating network array w/ eval_points
5. creating network array w/ multi-dimensional eval_points

"""

import math

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

# create the list of evaluation points
eval_points1 = np.arange(-1, 0, .5)
eval_points2 = np.array([[1,1],[-1,1],[-1,-1],[1,-1]]).T

net = nef.Network('EvalPoints Test')
net.make_input('in', value=math.sin)

# for test 1
net.make('A1', neurons=300, dimensions=1)
# for test 2
net.make('A2', neurons=300, dimensions=1, eval_points=eval_points1)
# for test 3
net.make('A3', neurons=300, dimensions=1, eval_points=eval_points1)
# for test 4
net.make('A4', neurons=300, array_size=3, dimensions=2, 
    eval_points=eval_points2)

net.make('B', neurons=100, dimensions=1)
net.make('C', neurons=100, dimensions=1)
net.make('D', neurons=100, dimensions=1)
net.make('E', neurons=100, dimensions=1)

# function for testing evaluation points
def pow(x):
    return [xval**2 for xval in x]

# create origins with eval_points
# for test 1
net.nodes['A1'].add_origin('pow', func=pow, eval_points=eval_points1)
# for test 3
net.nodes['A3'].add_origin('pow', func=pow, eval_points=eval_points1)

net.connect('in', 'A1')
net.connect('in', 'A2')
net.connect('in', 'A3')
net.connect('in', 'A4')
net.connect('A1:pow', 'B')  # for test 1
net.connect('A2', 'C', func=pow)  # for test 2
net.connect('A3:pow', 'D')  # for test 3
net.connect('A4', 'E', func=pow)  # for test 4

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
A1p = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
A2p = net.make_probe('C', dt_sample=dt_step, pstc=pstc)
A3p = net.make_probe('D', dt_sample=dt_step, pstc=pstc)
A4p = net.make_probe('E', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ioff(); plt.clf(); 
plt.subplot(511); plt.title('Input')
plt.plot(Ip.get_data())
plt.subplot(512); plt.title('A1')
plt.plot(A1p.get_data())
plt.subplot(513); plt.title('A2')
plt.plot(A2p.get_data())
plt.subplot(514); plt.title('A3')
plt.plot(A3p.get_data())
plt.subplot(515); plt.title('A4')
plt.plot(A4p.get_data())
plt.tight_layout()
plt.show()
