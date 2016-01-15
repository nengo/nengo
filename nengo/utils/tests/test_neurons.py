from __future__ import print_function

import pytest

import numpy as np

import nengo
from nengo.dists import Choice
from nengo.processes import WhiteSignal
from nengo.utils.matplotlib import implot
from nengo.utils.neurons import rates_isi, rates_kernel
from nengo.utils.numpy import rms


def _test_rates(Simulator, rates, plt, seed):
    n = 100
    intercepts = np.linspace(-0.99, 0.99, n)

    model = nengo.Network(seed=seed)
    with model:
        model.config[nengo.Ensemble].max_rates = Choice([50])
        model.config[nengo.Ensemble].encoders = Choice([[1]])
        u = nengo.Node(output=WhiteSignal(2, high=5))
        a = nengo.Ensemble(n, 1,
                           intercepts=intercepts, neuron_type=nengo.LIFRate())
        b = nengo.Ensemble(n, 1,
                           intercepts=intercepts, neuron_type=nengo.LIF())
        nengo.Connection(u, a, synapse=0)
        nengo.Connection(u, b, synapse=0)
        up = nengo.Probe(u)
        ap = nengo.Probe(a.neurons)
        bp = nengo.Probe(b.neurons)

    with Simulator(model, seed=seed+1) as sim:
        sim.run(2.)

    t = sim.trange()
    x = sim.data[up]
    a_rates = sim.data[ap]
    spikes = sim.data[bp]
    b_rates = rates(t, spikes)

    ax = plt.subplot(411)
    plt.plot(t, x)
    ax = plt.subplot(412)
    implot(plt, t, intercepts, a_rates.T, ax=ax)
    ax.set_ylabel('intercept')
    ax = plt.subplot(413)
    implot(plt, t, intercepts, b_rates.T, ax=ax)
    ax.set_ylabel('intercept')
    ax = plt.subplot(414)
    implot(plt, t, intercepts, (b_rates - a_rates).T, ax=ax)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('intercept')

    tmask = (t > 0.1) & (t < 1.9)
    relative_rmse = rms(b_rates[tmask] - a_rates[tmask]) / rms(a_rates[tmask])
    return relative_rmse


def test_rates_isi(Simulator, plt, seed):
    pytest.importorskip('scipy')
    rel_rmse = _test_rates(Simulator, rates_isi, plt, seed)
    assert rel_rmse < 0.3


def test_rates_kernel(Simulator, plt, seed):
    rel_rmse = _test_rates(Simulator, rates_kernel, plt, seed)
    assert rel_rmse < 0.25


@pytest.mark.noassertions
def test_rates(Simulator, plt, seed, logger):
    pytest.importorskip('scipy')
    functions = [
        ('isi_zero', lambda t, s: rates_isi(
            t, s, midpoint=False, interp='zero')),
        ('isi_midzero', lambda t, s: rates_isi(
            t, s, midpoint=True, interp='zero')),
        ('isi_linear', lambda t, s: rates_isi(
            t, s, midpoint=False, interp='linear')),
        ('isi_midlinear', lambda t, s: rates_isi(
            t, s, midpoint=True, interp='linear')),
        ('kernel_expon', lambda t, s: rates_kernel(t, s, kind='expon')),
        ('kernel_gauss', lambda t, s: rates_kernel(t, s, kind='gauss')),
        ('kernel_expogauss', lambda t, s: rates_kernel(
            t, s, kind='expogauss')),
        ('kernel_alpha', lambda t, s: rates_kernel(t, s, kind='alpha')),
    ]

    for name, function in functions:
        rel_rmse = _test_rates(Simulator, function, plt, seed)
        logger.info('rate estimator: %s', name)
        logger.info('relative RMSE: %0.4f', rel_rmse)
    plt.saveas = None
