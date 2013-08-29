import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.nonlinearities import PES

N = 30
D = 2

model = nengo.Model('Learn Communication')

# Create ensembles
model.make_ensemble('Pre', nengo.LIF(N * D), dimensions=D)
model.make_ensemble('Post', nengo.LIF(N * D), dimensions=D)
error = model.make_ensemble('Error', nengo.LIF(N * D), dimensions=D)

# Create an input signal
model.make_node('Input', output=lambda t: [np.sin(t), np.cos(t)])

model.connect('Input', 'Pre')

# Set the modulatory signal.
model.connect('Pre', 'Error')
model.connect('Post', 'Error', transform=np.eye(D) * -1)

# Create a modulated connection between the 'pre' and 'post' ensembles
model.connect('Pre', 'Post', function=lambda x: -1 * np.ones(x.shape),
              learning_rule=PES(error))

# For testing purposes
model.make_ensemble('Actual error', nengo.LIF(N * D), dimensions=D)
model.connect('Pre','Actual error')
model.connect('Post','Actual error', transform=np.eye(D) * -1)

model.probe('Pre', filter=0.02)
model.probe('Post', filter=0.02)
model.probe('Actual error', filter=0.02)

sim = model.simulator()
sim.run(5)

# Plot results
t = sim.data(model.t)
plt.figure(figsize=(6,5))
plt.subplot(211)
plt.plot(t, sim.data('Pre'), label='Pre')
plt.plot(t, sim.data('Post'), label='Post')
plt.legend()
plt.subplot(212)
plt.plot(t, sim.data('Actual error'), label='Error')
plt.legend()
plt.tight_layout()
plt.savefig('learning.pdf')
