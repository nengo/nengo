import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.nonlinearities import PES

N = 30
D = 2

model = nengo.Model('Learn Communication')
with model:
    # Create ensembles
    pre = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)
    post = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)
    error = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)

    # Create an input signal
    driver = nengo.Node(output=lambda t: [np.sin(t), np.cos(t)])

    nengo.Connection(driver, pre)

    # Set the modulatory signal.
    nengo.Connection(pre, error)
    nengo.Connection(post, error, transform=np.eye(D) * -1)

    # Create a modulated connection between the 'pre' and 'post' ensembles
    nengo.Connection(pre, post,
                     function=lambda x: -1 * np.ones(x.shape),
                     learning_rule=PES(error))

    # For testing purposes
    # XXX : what is semantics of error vs. actual_error?
    actual_error = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)
    nengo.Connection(pre, actual_error)
    nengo.Connection(post, actual_error, transform=np.eye(D) * -1)

    nengo.Probe(pre, 'decoded_output', filter=0.02)
    nengo.Probe(post, 'decoded_output', filter=0.02)
    nengo.Probe(actual_error, 'decoded_output', filter=0.02)

sim = nengo.Simultor(model)
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
plt.show()
#plt.tight_layout()
#plt.savefig('learning.pdf')
