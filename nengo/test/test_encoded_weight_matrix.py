"""This is a test file to test fully specifying the weight matrix with 
the transform parameter in net.connect. 

Tests:
  1. connect with T = (array_size * neurons x array_size * neurons)
  2. connect with T = (array_size x neurons x array_size * neurons)
  3. connect with T = (array_size * neurons x array_size x neurons)
  4. connect with T = (array_size x neurons x array_size x neurons)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

neurons = 100
dimensions = 1
array_size = 3
inhib_scale = 10

net = nef.Network('WeightMatrix Test')
net.make_input('in1', math.sin, zero_after_time=2.5)
net.make('A', neurons=neurons, dimensions=dimensions, intercept=(.1, 1))
net.make('B', neurons=neurons, dimensions=dimensions) # for test 1
net.make('B2', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 2 
net.make('B3', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 3 
net.make('B4', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 4

# setup inhibitory scaling matrix
weight_matrix_1 = [[1] * neurons * 1] * neurons * 1 # for test 1
weight_matrix_2 = [[[1] * neurons] * 1] * neurons * array_size # for test 1
weight_matrix_3 = [[[1] * neurons * 1] * neurons] * array_size # for test 3 
weight_matrix_4 = [[[[1] * neurons] * 1] * neurons] * array_size # for test 4

# define our transform and connect up! 
net.connect('in1', 'A')
net.connect('A', 'B', transform=weight_matrix_1) # for test 1
net.connect('A', 'B2', transform=weight_matrix_2) # for test 2
net.connect('A', 'B3', transform=weight_matrix_3) # for test 3
net.connect('A', 'B4', transform=weight_matrix_4) # for test 4

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in1', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
B2p = net.make_probe('B2', dt_sample=dt_step, pstc=pstc)
B3p = net.make_probe('B3', dt_sample=dt_step, pstc=pstc)
B4p = net.make_probe('B4', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

plt.ioff(); plt.close(); 
plt.subplot(611); plt.title('Input1')
plt.plot(Ip.get_data()); 
plt.subplot(612); plt.title('A')
plt.plot(Ap.get_data())
plt.subplot(613); plt.title('B')
plt.plot(Bp.get_data())
plt.subplot(614); plt.title('B2')
plt.plot(B2p.get_data())
plt.subplot(615); plt.title('B3')
plt.plot(B3p.get_data())
plt.subplot(616); plt.title('B4')
plt.plot(B4p.get_data())
plt.tight_layout()
plt.show()
