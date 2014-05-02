import logging

import numpy as np
import pytest

import nengo
from nengo.utils.functions import whitenoise
from nengo.utils.numpy import filt, lti
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def run_synapse(Simulator, synapse, dt=1e-3, runtime=1., n_neurons=None):
    model = nengo.Network()
    with model:
        u = nengo.Node(output=whitenoise(0.1, 5))

        if n_neurons is not None:
            a = nengo.Ensemble(n_neurons, 1)
            nengo.Connection(u, a, synapse=None)
            target = a
        else:
            target = u

        ref = nengo.Probe(target, synapse=None)
        filtered = nengo.Probe(target, synapse=synapse)

    sim = nengo.Simulator(model, dt=dt)
    sim.run(runtime)

    return sim.trange(), sim.data[ref], sim.data[filtered]


def test_lowpass(Simulator):
    dt = 1e-3
    tau = 0.03

    t, x, yhat = run_synapse(Simulator, nengo.synapses.Lowpass(tau), dt=dt)
    y = filt(x, tau / dt)

    with Plotter(Simulator) as plt:
        plt.plot(t, y)
        plt.plot(t, yhat, '--')
        plt.savefig('test_synapse.test_lowpass.pdf')
        plt.close()

    assert np.allclose(y[:-1], yhat[1:])
    # assert np.allclose(y, yhat)


def test_decoders(Simulator, nl):
    dt = 1e-3
    tau = 0.01

    t, x, yhat = run_synapse(
        Simulator, nengo.synapses.Lowpass(tau), dt=dt, n_neurons=100)

    y = filt(x, tau / dt)
    assert np.allclose(y[:-1], yhat[1:])


@pytest.mark.optional
def test_general(Simulator):
    import scipy.signal

    dt = 1e-3
    order = 4
    tau = 0.03

    num, den = scipy.signal.butter(order, 1. / tau, analog=True)
    numi, deni, dt = scipy.signal.cont2discrete((num, den), dt)

    t, x, yhat = run_synapse(
        Simulator, nengo.synapses.LinearFilter(num, den), dt=dt)
    y = lti(x, (numi, deni))

    with Plotter(Simulator) as plt:
        plt.plot(t, x)
        plt.plot(t, y)
        plt.plot(t, yhat, '--')
        plt.savefig('test_synapse.test_general.pdf')
        plt.close()

    assert np.allclose(y, yhat)
