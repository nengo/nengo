import numpy as np

import nengo
from nengo.utils.compat import range
from nengo.utils.numpy import rmse


def test_sine_waves(Simulator, plt, seed):
    radius = 2
    dim = 5
    product = nengo.networks.Product(
        200, dim, radius, net=nengo.Network(seed=seed))

    func_A = lambda t: np.sqrt(radius)*np.sin(np.arange(1, dim+1)*2*np.pi*t)
    func_B = lambda t: np.sqrt(radius)*np.sin(np.arange(dim, 0, -1)*2*np.pi*t)
    with product:
        input_A = nengo.Node(func_A)
        input_B = nengo.Node(func_B)
        nengo.Connection(input_A, product.A)
        nengo.Connection(input_B, product.B)
        p = nengo.Probe(product.output, synapse=0.005)

    sim = Simulator(product)
    sim.run(1.0)

    t = sim.trange()
    AB = np.asarray(list(map(func_A, t))) * np.asarray(list(map(func_B, t)))
    delay = 0.013
    offset = np.where(t >= delay)[0]

    for i in range(dim):
        plt.subplot(dim+1, 1, i+1)
        plt.plot(t + delay, AB[:, i])
        plt.plot(t, sim.data[p][:, i])
    plt.xlim(right=t[-1])

    assert rmse(AB[:len(offset), :], sim.data[p][offset, :]) < 0.2
