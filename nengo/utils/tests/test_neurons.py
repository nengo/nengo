from __future__ import print_function

import logging
import pytest

import numpy as np

import nengo
from nengo.utils.functions import whitenoise
from nengo.utils.matplotlib import implot
from nengo.utils.neurons import rates_isi, rates_kernel
from nengo.utils.numpy import rms
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def _test_rates(Simulator, rates, name=None):
    if name is None:
        name = rates.__name__

    n = 100
    max_rates = 50 * np.ones(n)
    # max_rates = 200 * np.ones(n)
    intercepts = np.linspace(-0.99, 0.99, n)
    encoders = np.ones((n, 1))
    nparams = dict(n_neurons=n)
    eparams = dict(
        max_rates=max_rates, intercepts=intercepts, encoders=encoders)

    model = nengo.Network()
    with model:
        u = nengo.Node(output=whitenoise(1, 5, seed=8393))
        a = nengo.Ensemble(nengo.LIFRate(**nparams), 1, **eparams)
        b = nengo.Ensemble(nengo.LIF(**nparams), 1, **eparams)
        nengo.Connection(u, a, synapse=0)
        nengo.Connection(u, b, synapse=0)
        up = nengo.Probe(u)
        ap = nengo.Probe(a.neurons, "output", filter=0)
        bp = nengo.Probe(b.neurons, "output", filter=0)

    dt = 1e-3
    sim = Simulator(model, dt=dt)
    sim.run(2.)

    t = sim.trange()
    x = sim.data[up]
    a_rates = sim.data[ap] / dt
    spikes = sim.data[bp]
    b_rates = rates(t, spikes)

    with Plotter(Simulator) as plt:
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
        plt.savefig('utils.test_neurons.test_rates.%s.pdf' % name)
        plt.close()

    tmask = (t > 0.1) & (t < 1.9)
    relative_rmse = rms(b_rates[tmask] - a_rates[tmask]) / rms(a_rates[tmask])
    return relative_rmse


@pytest.mark.optional  # requires Scipy
def test_rates_isi(Simulator):
    rel_rmse = _test_rates(Simulator, rates_isi)
    assert rel_rmse < 0.3


def test_rates_kernel(Simulator):
    rel_rmse = _test_rates(Simulator, rates_kernel)
    assert rel_rmse < 0.2


@pytest.mark.benchmark
def test_rates(Simulator):
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

    print("\ntest_rates:")
    for name, function in functions:
        rel_rmse = _test_rates(Simulator, function, name)
        print("%20s relative rmse: %0.3f" % (name, rel_rmse))


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
