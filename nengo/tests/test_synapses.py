import logging

import pytest

import nengo
from nengo.utils.functions import whitenoise
from nengo.utils.numpy import filt, lti
from nengo.utils.testing import Plotter, allclose

logger = logging.getLogger(__name__)


def run_synapse(Simulator, synapse, dt=1e-3, runtime=1., n_neurons=None):
    model = nengo.Network(seed=2984)
    with model:
        u = nengo.Node(output=whitenoise(0.1, 5, seed=328))

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

    assert allclose(t, y.flatten(), yhat.flatten(), delay=1,
                    plotter=Plotter(Simulator),
                    filename='test_synapse.test_lowpass.pdf')


@pytest.mark.optional
def test_alpha(Simulator):
    dt = 1e-3
    tau = 0.03
    b, a = [0.00054336, 0.00053142], [1, -1.9344322, 0.93550699]
    # ^^^ these coefficients found for tau=0.03 and dt=1e-3
    #   scipy.signal.cont2discrete(([1], [tau**2, 2*tau, 1]), dt)

    # b = [0.00054336283526056767, 0.00053142123234546667]
    # a = [1, -1.9344322009640118, 0.93550698503161778]
    # ^^^ these coefficients found by the exact algorithm used in Builder

    t, x, yhat = run_synapse(Simulator, nengo.synapses.Alpha(tau), dt=dt)
    y = lti(x, (b, a))

    assert allclose(t, y.flatten(), yhat.flatten(), delay=1, atol=5e-6,
                    plotter=Plotter(Simulator),
                    filename='test_synapse.test_alpha.pdf')


def test_decoders(Simulator, nl):
    dt = 1e-3
    tau = 0.01

    t, x, yhat = run_synapse(
        Simulator, nengo.synapses.Lowpass(tau), dt=dt, n_neurons=100)

    y = filt(x, tau / dt)
    assert allclose(t, y.flatten(), yhat.flatten(), delay=1,
                    plotter=Plotter(Simulator, nl),
                    filename='test_synapse.test_decoders.pdf')


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

    assert allclose(t, y.flatten(), yhat.flatten(),
                    plotter=Plotter(Simulator),
                    filename='test_synapse.test_general.pdf')
