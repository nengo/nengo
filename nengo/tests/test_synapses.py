import numpy as np
import pytest

import nengo
from nengo.processes import WhiteSignal
from nengo.synapses import (
    Alpha, filt, filtfilt, LinearFilter, Lowpass, SynapseParam, Triangle)
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

    sim = Simulator(model, dt=dt, seed=seed+1)
    sim.run(runtime)

    return sim.trange(), sim.data[ref], sim.data[filtered]


def test_lowpass(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.03

    t, x, yhat = run_synapse(Simulator, seed, Lowpass(tau), dt=dt)
    y = filt(x, tau, dt=dt)

    assert allclose(t, y, yhat, delay=dt, plt=plt)


def test_alpha(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.03
    num, den = [1], [tau**2, 2*tau, 1]

    t, x, yhat = run_synapse(Simulator, seed, Alpha(tau), dt=dt)
    y = filt(x, LinearFilter(num, den), dt=dt)

    assert allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt)


def test_triangle(Simulator, plt, seed):
    dt = 1e-3
    tau = 0.03

    t, x, ysim = run_synapse(Simulator, seed, Triangle(tau), dt=dt)
    yfilt = filt(x, Triangle(tau), dt=dt)

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

    y = filt(x, tau, dt=dt)
    assert allclose(t, y, yhat, delay=dt, plt=plt)


def test_linearfilter(Simulator, plt, seed):
    dt = 1e-3

    # The following num, den are for a 4th order analog Butterworth filter,
    # generated with `scipy.signal.butter(4, 1. / 0.03, analog=True)`
    num = np.array([1234567.90123457])
    den = np.array([1.0, 87.104197658425107, 3793.5706248589954,
                    96782.441842694592, 1234567.9012345686])

    t, x, yhat = run_synapse(Simulator, seed, LinearFilter(num, den), dt=dt)
    y = filt(x, LinearFilter(num, den), dt=dt)

    assert allclose(t, y, yhat, delay=dt, plt=plt)


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

    y = filt(u, 0.1, dt=dt)

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
    x = filt(u, tau, dt=dt)
    x = filt(x[::-1], tau, x0=x[-1], dt=dt)[::-1]
    y = filtfilt(u, tau, dt=dt)

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
    x = filt(u, tau, dt=dt)
    y = filt(u, lti, dt=dt)

    plt.plot(t, x[:, 0], label="Lowpass")
    plt.plot(t, y[:, 0], label="LTI")
    plt.legend(loc="best")

    assert np.allclose(x, y)


def test_synapseparam():
    """SynapseParam must be a Synapse, and converts numbers to LowPass."""
    class Test(object):
        sp = SynapseParam(default=Lowpass(0.1))

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
