"""
Nengo Example: Controlled Integrator

A controlled integrator is a circuit that acts on two signals:

1. Input - the signal being integrated
2. Control - the control signal to the integrator

A controlled integrator accumulates input, but its state can be directly manipulated by the control signal.
We can write the dynamics of a simple controlled integrator like this:

$$
\dot{a}(t) = \mathrm{control}(t) \cdot a(t) + B \cdot \mathrm{input}(t)
$$

In this notebook, we will build a controlled intgrator with Leaky Integrate and Fire ([LIF](TODO)) neurons.
The Neural Engineering Framework ([NEF](TODO)) equivalent equation for this integrator is:

$$
\dot{a}(t) = \mathrm{control}(t) \cdot a(t) + \tau \cdot \mathrm{input}(t).
$$

We call the coefficient $\tau$ here a *recurrent time constant* because it governs the rate of integration.

Network behaviour:
`A = tau * Input + Input * Control`

(MISSING: Network circuit diagram - can maybe generate this in-line?)
"""

### do some setup before we start
import numpy as np
import matplotlib.pyplot as plt

import nengo
model = nengo.Model('Controlled Integrator')
print "seed", model.seed

# Make a population with 225 LIF neurons
# representing a 2 dimensional signal,
# with a larger radius to accommodate large inputs
A = model.make_ensemble('A', nengo.LIF(225), dimensions=2, radius=1.5)

plt.figure(3)
plt.clf()
x = np.linspace(-1, 1, 101)
acts = A.neurons.rates(A.neurons.gain[None,:] * x[:,None])
plt.plot(x, acts)

def input_func(t):              # Create a function that outputs
    if   t < 0.2:  return 0     # 0 for t < 0.2
    elif t < 0.3:  return 5     # 5 for 0.2 < t < 0.3
    elif t < 0.44: return 0     # 0 for 0.3 < t < 0.44
    elif t < 0.54: return -10   # -10 for 0.44 < t < 0.54
    elif t < 0.8:  return 0     # 0 for 0.54 < t < 0.8
    elif t < 0.9:  return 5     # 5 for 0.8 < t < 0.9
    else:          return 0     # 0 for t > 0.9

t = np.linspace(0, 1, 101)
plt.figure(1)
plt.clf()
plt.plot(t, map(input_func, t))
plt.ylim((-11,11));

model.make_node('Input', output=input_func)

tau = 0.1
model.connect('Input', 'A',
              transform=[[tau], [0]],
              filter=tau)

def control_func(t):            # Create a function that outputs
    if   t < 0.6: return 1      # 1 for t < 0.65
    else:         return 0.5    # 0.5 for t > 0.65

t = np.linspace(0, 1, 101)
plt.plot(t, map(control_func, t))
plt.ylim(0,1.1);

model.make_node('Control', output=control_func)
model.connect('Control', 'A', transform=[[0], [1]], filter=0.005)

model.connect('A', 'A',
              function=lambda x: x[0] * x[1],  # -- function is applied first to A
              transform=[[1], [0]],            # -- transform converts function output to new state inputs
              filter=tau)

# Record both dimensions of A
model.probe('A', filter=0.02)

# Run the model for 1.4 seconds
model.run(1.4)

# Plot the value and control signals, along with the exact integral
t = model.data[model.simtime]
dt = t[1] - t[0]
input_sig = map(input_func, t)
control_sig = map(control_func, t)
ref = dt * np.cumsum(input_sig)

plt.figure(2, figsize=(6,8))
plt.clf()
plt.subplot(211)
plt.plot(t, input_sig, label='input')
plt.ylim(-11, 11)
plt.ylabel('input')
plt.legend(loc=3, frameon=False)

plt.subplot(212)
plt.plot(t, ref, 'k--', label='exact')
plt.plot(t, model.data['A'][:,0], label='A (value)')
plt.plot(t, model.data['A'][:,1], label='A (control)')
plt.ylim([-1.1, 1.1])
plt.xlabel('time [s]')
plt.ylabel('x(t)')
plt.legend(loc=3, frameon=False);

plt.show()
