import numpy as np
import matplotlib.pyplot as plt

import nengo

N = 30
D = 2

model = nengo.Model('Learn Communication')
with model:
    # Create ensembles
    pre = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)
    post = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)
    error = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)

    # Create an input signal
    driver = nengo.Node(output=lambda t: [np.sin(3 * t), np.cos(3 * t)])
    lr_node = nengo.Node(output=lambda t: 1.0 if t < 5 else 0)

    nengo.Connection(driver, pre)

    # Set the modulatory signal.
    nengo.Connection(pre, error)
    nengo.Connection(post, error, transform=-1)

    # Create a modulated connection between the 'pre' and 'post' ensembles
    pes_rule = nengo.PES_Rule(
        nengo.Connection(pre, post,
                         function=lambda x: -1 * np.ones(x.shape)),
        error=error,
        base_learning_rate=0.0)
    nengo.Connection(lr_node, pes_rule.learning_rate)

    # For testing purposes
    # XXX : what is semantics of error vs. actual_error?
    actual_error = nengo.Ensemble(nengo.LIF(N * D), dimensions=D)
    nengo.Connection(pre, actual_error)
    nengo.Connection(post, actual_error, transform=np.eye(D) * -1)

    p_pre = nengo.Probe(pre, 'decoded_output', filter=0.02)
    p_post = nengo.Probe(post, 'decoded_output', filter=0.02)
    p_aerr = nengo.Probe(actual_error, 'decoded_output', filter=0.02)
    p_lr = nengo.Probe(lr_node, 'output', filter=0.02)

sim = nengo.Simulator(model)
plt.figure(figsize=(6, 5))
sim.run(10)
# Plot results
t = sim.trange()
plt.subplot(211)
plt.plot(t, sim.data(p_pre), label='Pre')
plt.plot(t, sim.data(p_post), label='Post')
plt.legend()
plt.subplot(212)
plt.plot(t, sim.data(p_aerr), label='Error')
plt.plot(t, sim.data(p_lr), label='Learning Rate')
plt.legend()
plt.tight_layout()
plt.savefig('learning.pdf')
