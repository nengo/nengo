import numpy as np
import pytest

import nengo
from nengo.processes import WhiteSignal
from nengo.synapses import (
    Alpha, LinearFilter, Lowpass, SynapseParam, Triangle)
from nengo.utils.testing import allclose


def run_synapse(Simulator, seed, synapse, dt=1e-3, runtime=1., n_neurons=None):
    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(output=WhiteSignal(runtime, 5))

        if n_neurons is not None:
            a = nengo.Ensemble(n_neurons, 1)
            nengo.Connection(u, a, synapse=None)
            target = a
        else:
            target = u

        ref = nengo.Probe(target)
        filtered = nengo.Probe(target, synapse=synapse)

    with Simulator(model, dt=dt, seed=seed+1) as sim:
        sim.run(runtime)

    return sim.trange(), sim.data[ref], sim.data[filtered]


def test_lowpass(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.03

    t, x, yhat = run_synapse(Simulator, seed, Lowpass(tau), dt=dt)
    y = Lowpass(tau).filt(x, dt=dt, y0=0)

    assert allclose(t, y, yhat, delay=dt, plt=plt)


def test_alpha(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.03
    num, den = [1], [tau**2, 2*tau, 1]

    t, x, yhat = run_synapse(Simulator, seed, Alpha(tau), dt=dt)
    y = LinearFilter(num, den).filt(x, dt=dt, y0=0)

    assert allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt)


def test_triangle(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.03

    t, x, ysim = run_synapse(Simulator, seed, Triangle(tau), dt=dt)
    yfilt = Triangle(tau).filt(x, dt=dt, y0=0)

    # compare with convolved filter
    n_taps = int(round(tau / dt)) + 1
    num = np.arange(n_taps, 0, -1, dtype=float)
    num /= num.sum()
    y = np.convolve(x.ravel(), num)[:len(t)]
    y.shape = (-1, 1)

    assert np.allclose(y, yfilt, rtol=0)
    assert allclose(t, y, ysim, delay=dt, rtol=0, plt=plt)


def test_decoders(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.01

    t, x, yhat = run_synapse(
        Simulator, seed, Lowpass(tau), dt=dt, n_neurons=100)

    y = Lowpass(tau).filt(x, dt=dt, y0=0)
    assert allclose(t, y, yhat, delay=dt, plt=plt)


def test_linearfilter(Simulator, plt, seed):
    dt = 1e-3

    # The following num, den are for a 4th order analog Butterworth filter,
    # generated with `scipy.signal.butter(4, 0.2, analog=False)`
    num = np.array(
        [0.00482434, 0.01929737, 0.02894606, 0.01929737, 0.00482434])
    den = np.array([1., -2.36951301,  2.31398841, -1.05466541,  0.18737949])

    synapse = LinearFilter(num, den, analog=False)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert allclose(t, y, yhat, delay=dt, plt=plt)


def test_step_errors():
    output = np.zeros(3)
    with pytest.raises(ValueError):
        LinearFilter.NoDen([1], [1], output)
    with pytest.raises(ValueError):
        LinearFilter.Simple([1, 2], [1], output)
    with pytest.raises(ValueError):
        LinearFilter.Simple([1], [1, 2], output)


def test_filt(plt, rng):
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 0.1 / dt

    u = rng.normal(size=nt)

    tk = np.arange(0, 30 * tau)
    k = 1. / tau * np.exp(-tk / tau)
    x = np.convolve(u, k, mode='full')[:nt]

    y = Lowpass(0.1).filt(u, dt=dt, y0=0)

    plt.plot(t, x)
    plt.plot(t, y, '--')

    assert np.allclose(x, y, atol=1e-3, rtol=1e-2)


def test_filtfilt(plt, rng):
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 0.03

    u = rng.normal(size=nt)
    x = Lowpass(tau).filt(u, dt=dt)
    x = Lowpass(tau).filt(x[::-1], y0=x[-1], dt=dt)[::-1]
    y = Lowpass(tau).filtfilt(u, dt=dt)

    plt.plot(t, x)
    plt.plot(t, y, '--')

    assert np.allclose(x, y)


def test_lti_lowpass(rng, plt):
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2
    lti = LinearFilter([1], [tau, 1])

    u = rng.normal(size=(nt, 10))
    x = Lowpass(tau).filt(u, dt=dt)
    y = lti.filt(u, dt=dt)

    plt.plot(t, x[:, 0], label="Lowpass")
    plt.plot(t, y[:, 0], label="LTI")
    plt.legend(loc="best")

    assert np.allclose(x, y)


def test_synapseparam():
    """SynapseParam must be a Synapse, and converts numbers to LowPass."""
    class Test(object):
        sp = SynapseParam('sp', default=Lowpass(0.1))

    inst = Test()
    assert isinstance(inst.sp, Lowpass)
    assert inst.sp.tau == 0.1
    # Number are converted to LowPass
    inst.sp = 0.05
    assert isinstance(inst.sp, Lowpass)
    assert inst.sp.tau == 0.05
    # None has meaning
    inst.sp = None
    assert inst.sp is None
    # Non-synapse not OK
    with pytest.raises(ValueError):
        inst.sp = 'a'


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = LinearFilter([1], [0.04, 1])
    b = LinearFilter([1], [0.04, 1])
    c = LinearFilter([1], [0.04, 1.1])

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely

    with pytest.raises((ValueError, RuntimeError)):
        a.den[0] = 9
