import numpy as np
import pytest

import nengo
from nengo.utils.functions import HilbertCurve
from nengo.utils.numpy import rmse, maxint


def test_sine_waves(Simulator, plt, seed):
    radius = 2
    dim = 5
    product = nengo.networks.Product(200, dim, radius, seed=seed)

    func_a = lambda t: np.sqrt(radius)*np.sin(np.arange(1, dim+1)*2*np.pi*t)
    func_b = lambda t: np.sqrt(radius)*np.sin(np.arange(dim, 0, -1)*2*np.pi*t)
    with product:
        input_a = nengo.Node(func_a)
        input_b = nengo.Node(func_b)
        nengo.Connection(input_a, product.input_a)
        nengo.Connection(input_b, product.input_b)
        p = nengo.Probe(product.output, synapse=0.005)

    with Simulator(product) as sim:
        sim.run(1.0)

    t = sim.trange()
    ideal = (np.asarray([func_a(tt) for tt in t])
             * np.asarray([func_b(tt) for tt in t]))
    delay = 0.013
    offset = np.where(t >= delay)[0]

    for i in range(dim):
        plt.subplot(dim+1, 1, i+1)
        plt.plot(t + delay, ideal[:, i])
        plt.plot(t, sim.data[p][:, i])
        plt.xlim(right=t[-1])
        plt.yticks((-2, 0, 2))

    assert rmse(ideal[:len(offset), :], sim.data[p][offset, :]) < 0.2


def test_direct_mode_with_single_neuron(Simulator, plt, seed):
    radius = 2
    dim = 5

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        product = nengo.networks.Product(1, dim, radius, seed=seed)

    func_a = lambda t: np.sqrt(radius)*np.sin(np.arange(1, dim+1)*2*np.pi*t)
    func_b = lambda t: np.sqrt(radius)*np.sin(np.arange(dim, 0, -1)*2*np.pi*t)
    with product:
        input_a = nengo.Node(func_a)
        input_b = nengo.Node(func_b)
        nengo.Connection(input_a, product.input_a)
        nengo.Connection(input_b, product.input_b)
        p = nengo.Probe(product.output, synapse=0.005)

    with Simulator(product) as sim:
        sim.run(1.0)

    t = sim.trange()
    ideal = (np.asarray([func_a(tt) for tt in t])
             * np.asarray([func_b(tt) for tt in t]))
    delay = 0.013
    offset = np.where(t >= delay)[0]

    for i in range(dim):
        plt.subplot(dim+1, 1, i+1)
        plt.plot(t + delay, ideal[:, i])
        plt.plot(t, sim.data[p][:, i])
        plt.xlim(right=t[-1])
        plt.yticks((-2, 0, 2))

    assert rmse(ideal[:len(offset), :], sim.data[p][offset, :]) < 0.2


@pytest.mark.benchmark
@pytest.mark.slow
def test_product_benchmark(Simulator, analytics, rng):
    n_trials = 50
    hc = HilbertCurve(n=4)  # Increase n to cover the input space more densely
    duration = 5.           # Simulation duration (s)
    # Duration (s) to wait at the beginning to have a stable representation
    wait_duration = 0.5
    n_neurons = 100
    n_eval_points = 1000

    def stimulus_fn(t):
        return np.squeeze(hc(t / duration).T * 2 - 1)

    def run_trial():
        model = nengo.Network(seed=rng.randint(maxint))
        with model:
            model.config[nengo.Ensemble].n_eval_points = n_eval_points

            stimulus = nengo.Node(
                output=lambda t: stimulus_fn(max(0., t - wait_duration)),
                size_out=2)

            product_net = nengo.networks.Product(n_neurons, 1)
            nengo.Connection(stimulus[0], product_net.input_a)
            nengo.Connection(stimulus[1], product_net.input_b)
            probe_test = nengo.Probe(product_net.output)

            ens_direct = nengo.Ensemble(
                1, dimensions=2, neuron_type=nengo.Direct())
            result_direct = nengo.Node(size_in=1)
            nengo.Connection(stimulus, ens_direct)
            nengo.Connection(
                ens_direct, result_direct, function=lambda x: x[0] * x[1],
                synapse=None)
            probe_direct = nengo.Probe(result_direct)

        with Simulator(model) as sim:
            sim.run(duration + wait_duration, progress_bar=False)

        selection = sim.trange() > wait_duration
        test = sim.data[probe_test][selection]
        direct = sim.data[probe_direct][selection]
        return rmse(test, direct)

    error_data = [run_trial() for i in range(n_trials)]
    analytics.add_data(
        'error', error_data, "Multiplication RMSE. Shape: n_trials")


@pytest.mark.compare
def test_compare_product_benchmark(analytics_data, logger):
    stats = pytest.importorskip('scipy.stats')
    data1, data2 = (d['error'] for d in analytics_data)
    improvement = np.mean(data1) - np.mean(data2)
    p = np.ceil(1000. * 2. * stats.mannwhitneyu(
        data1, data2, alternative='two-sided')[1]) / 1000.
    logger.info("Multiplication improvement by %f (%.0f%%, p < %.3f)",
                improvement, (1. - np.mean(data2) / np.mean(data1)) * 100., p)
    assert improvement >= 0. or p >= 0.05
