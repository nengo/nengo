"""
This example demonstrates how to create an integrator in neurons.
  The function an integrator implements can be written in the
  following control theoretic equation:

    a_dot(t) = A * a(t) + B * input(t)

  The NEF equivalent equation for this integrator is:

    a_dot(t) = a(t) + tau * input(t)

  where tau is the recurrent time constant.

Network diagram:

                    .----.
                    v    |
     [Input] ----> (A) --'


Network behaviour:
   A = tau * Input + Input

"""

# from .. import nengo as nengo
# from ..nengo.connection import gen_transfrom
# from ..nengo.filter import ExponentialPSC

import collections

import nengo.old_api as nengo

# Define model parameters
tau = 0.1

# Create the nengo model
model = nengo.Network('Integrator')

# Create the model inputs
input_d = {0.2:5, 0.3:0, 0.44:-10, 0.54:0, 0.8:5, 0.9:0}
def input_f(t):
    # for key, value in input_d.items():
    #     if t < key:
    #         return value
    # return 0.0
    # t = t/10
    # if t < 0.2: return 5
    # elif t < 0.3: return 0
    # elif t < 0.44: return -10
    # elif t < 0.54: return 0
    # elif t < 0.8: return 5
    # elif t < 0.9: return 0
    if t < 0.7:
        return 1
    else:
        return 0

model.make_input('Input', input_f)
# model.make_input('Input', input_d)

# Create the neuronal ensembles
model.make('A', 100, 1)

# Create the connections within the model
model.connect('Input', 'A', transform=tau, pstc=0.1)

model.connect('A', 'A', pstc=tau)

# Add probes
dt_sample = 0.01
in_p = model.make_probe('Input', dt_sample, dt_sample)
out_p = model.make_probe('A', dt_sample, dt_sample)

# model.connect('Input', 'in_p')
# model.connect('A', 'out_p')

# Build the model
# model.build()

# Run the model
model.run(3)

# print in_p

# import ipdb
# ipdb.set_trace()

import numpy as np
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
x = in_p.get_data()
y = out_p.get_data()
t = dt_sample*np.arange(len(x))
plt.plot(t, x)
plt.plot(t, y)
plt.show()
