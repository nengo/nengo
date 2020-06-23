import numpy as np

import nengo
from nengo.networks.product import dot_product_transform
from nengo.utils.numpy import rms


def test_dot_product_transform():
    """tests the function dot_product_transform"""
    assert dot_product_transform(2).all() == 1


def test_sine_waves(Simulator, plt, seed):
    radius = 2
    dim = 5
    product = nengo.networks.Product(200, dim, radius, seed=seed)

    func_a = lambda t: np.sqrt(radius) * np.sin(np.arange(1, dim + 1) * 2 * np.pi * t)
    func_b = lambda t: np.sqrt(radius) * np.sin(np.arange(dim, 0, -1) * 2 * np.pi * t)
    with product:
        input_a = nengo.Node(func_a)
        input_b = nengo.Node(func_b)
        nengo.Connection(input_a, product.input_a)
        nengo.Connection(input_b, product.input_b)
        p = nengo.Probe(product.output, synapse=0.005)

    with Simulator(product) as sim:
        sim.run(1.0)

    t = sim.trange()
    ideal = np.asarray([func_a(tt) for tt in t]) * np.asarray([func_b(tt) for tt in t])
    delay = 0.013
    offset = np.where(t >= delay)[0]

    for i in range(dim):
        plt.subplot(dim + 1, 1, i + 1)
        plt.plot(t + delay, ideal[:, i])
        plt.plot(t, sim.data[p][:, i])
        plt.xlim(right=t[-1])
        plt.yticks((-2, 0, 2))

    assert rms(ideal[: len(offset), :] - sim.data[p][offset, :]) < 0.2


def test_direct_mode_with_single_neuron(Simulator, plt, seed):
    radius = 2
    dim = 5

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        product = nengo.networks.Product(1, dim, radius, seed=seed)

    func_a = lambda t: np.sqrt(radius) * np.sin(np.arange(1, dim + 1) * 2 * np.pi * t)
    func_b = lambda t: np.sqrt(radius) * np.sin(np.arange(dim, 0, -1) * 2 * np.pi * t)
    with product:
        input_a = nengo.Node(func_a)
        input_b = nengo.Node(func_b)
        nengo.Connection(input_a, product.input_a)
        nengo.Connection(input_b, product.input_b)
        p = nengo.Probe(product.output, synapse=0.005)

    with Simulator(product) as sim:
        sim.run(1.0)

    t = sim.trange()
    ideal = np.asarray([func_a(tt) for tt in t]) * np.asarray([func_b(tt) for tt in t])
    delay = 0.013
    offset = np.where(t >= delay)[0]

    for i in range(dim):
        plt.subplot(dim + 1, 1, i + 1)
        plt.plot(t + delay, ideal[:, i])
        plt.plot(t, sim.data[p][:, i])
        plt.xlim(right=t[-1])
        plt.yticks((-2, 0, 2))

    assert rms(ideal[: len(offset), :] - sim.data[p][offset, :]) < 0.2
