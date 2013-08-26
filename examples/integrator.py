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

import nengo

# Define model parameters
tau = 0.1      # Recurrent time constant

# Create the nengo model
model = nengo.Model('Integrator')
print model.seed

# Create the model inputs
# {0.2:5, 0.3:0, 0.44:-10, 0.54:0, 0.8:5, 0.9:0}
def input_func(t):              # Create a function that outputs
    if   t < 0.2:  return 0     # 0 for t < 0.2
    elif t < 0.3:  return 5     # 5 for 0.2 < t < 0.3
    elif t < 0.44: return 0     # 0 for 0.3 < t < 0.44
    elif t < 0.54: return -10   # -10 for 0.44 < t < 0.54
    elif t < 0.8:  return 0     # 0 for 0.54 < t < 0.8
    elif t < 0.9:  return 5     # 5 for 0.8 < t < 0.9
    else:          return 0     # 0 for t > 0.9

model.make_node('Input', input_func)

# A = model.make_ensemble('A', 100, 1)
# model.connect('Input', 'A', transform=tau, filter=0.1)
# model.connect('A', 'A', filter=tau)

model.make_node('Control', [1])
A = model.make_ensemble('A', 500, 2, radius=1.5)
model.connect('Input', 'A', transform=[[tau], [0]], filter=0.1)
model.connect('Control', 'A', transform=[[0],[1]])
model.connect('A', 'A', function=lambda x: x[0]*x[1], transform=[[1],[0]], filter=tau)


model.probe('A', filter=0.02)

# Build the model
# model.build()

# Run the model
model.run(1)

### Plotting
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

t = model.data[model.simtime]
dt = t[1] - t[0]

input_sig = map(input_func, t)
exact_sig = dt*np.cumsum(input_sig)

plt.figure(1)
plt.clf()
plt.plot(t, exact_sig, 'k--')
plt.plot(t, model.data['A'])
