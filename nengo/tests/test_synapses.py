from inspect import getfullargspec

import numpy as np
from numpy import array  # pylint: disable=unused-import
import pytest

import nengo
from nengo.exceptions import ValidationError
from nengo.processes import WhiteSignal
from nengo.synapses import (
    Alpha, LinearFilter, Lowpass, Synapse, SynapseParam, Triangle)
from nengo.utils.filter_design import cont2discrete
from nengo.utils.testing import signals_allclose


# The following num, den are for a 4th order analog Butterworth filter,
# generated with `scipy.signal.butter(4, 0.1, analog=False)`
butter_num = np.array([0.0004166, 0.0016664, 0.0024996, 0.0016664, 0.0004166])
butter_den = np.array([1., -3.18063855, 3.86119435, -2.11215536, 0.43826514])


def run_synapse(Simulator, seed, synapse, dt=1e-3, runtime=0.2, high=100,
                n_neurons=None):
    model = nengo.Network(seed=seed)
    with model:
        u = nengo.Node(output=WhiteSignal(runtime, high=high))

        if n_neurons is not None:
            a = nengo.Ensemble(n_neurons, 1)
            nengo.Connection(u, a, synapse=None)
            target = a
        else:
            target = u

        ref = nengo.Probe(target)
        filtered = nengo.Probe(target, synapse=synapse)

    with Simulator(model, dt=dt, seed=seed + 1) as sim:
        sim.run(runtime)

    return sim.trange(), sim.data[ref], sim.data[filtered]


def test_direct(Simulator, plt, seed, allclose):
    dt = 1e-3
    a = 0.7

    synapse = LinearFilter([a], [1], analog=False)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, allclose=allclose)
    assert signals_allclose(t, a * x, y, plt=plt, allclose=allclose)


def test_lowpass(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.03

    t, x, yhat = run_synapse(Simulator, seed, Lowpass(tau), dt=dt)
    y = Lowpass(tau).filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, plt=plt, allclose=allclose)


def test_alpha(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.03
    num, den = [1], [tau ** 2, 2 * tau, 1]

    t, x, yhat = run_synapse(Simulator, seed, Alpha(tau), dt=dt)
    y = LinearFilter(num, den).filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt,
                            allclose=allclose)


def test_triangle(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.03

    t, x, ysim = run_synapse(Simulator, seed, Triangle(tau), dt=dt)
    yfilt = Triangle(tau).filt(x, dt=dt, y0=0)

    # compare with convolved filter
    n_taps = int(round(tau / dt)) + 1
    num = np.arange(n_taps, 0, -1, dtype=nengo.rc.float_dtype)
    num /= num.sum()
    y = np.convolve(x.ravel(), num)[:len(t)]
    y.shape = (-1, 1)

    assert allclose(y, yfilt, rtol=0)
    assert signals_allclose(t, y, ysim, delay=dt, rtol=0, plt=plt,
                            allclose=allclose)

    # test y0 != 0
    assert allclose(Triangle(tau).filt(np.ones(100), dt=dt, y0=1), 1)


def test_decoders(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.01

    t, x, yhat = run_synapse(
        Simulator, seed, Lowpass(tau), dt=dt, n_neurons=100)

    y = Lowpass(tau).filt(x, dt=dt, y0=0)
    assert signals_allclose(t, y, yhat, delay=dt, plt=plt,
                            allclose=allclose)


def test_linearfilter(Simulator, plt, seed, allclose):
    dt = 1e-3
    synapse = LinearFilter(butter_num, butter_den, analog=False)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, plt=plt, allclose=allclose)


def test_linearfilter_y0(allclose):
    # --- y0 sets initial state correctly for high-order filter
    synapse = LinearFilter(butter_num, butter_den, analog=False)
    v = 9.81
    x = v * np.ones(10)
    assert allclose(synapse.filt(x, y0=v), v)
    assert not allclose(synapse.filt(x, y0=0), v)

    # --- y0 does not work for high-order synapse when DC gain is zero
    synapse = LinearFilter([1, 0], [1, 1])
    with pytest.raises(ValidationError, match="Cannot solve for state"):
        synapse.filt(np.ones(10), y0=1)


def test_linearfilter_extras(allclose):
    # This filter is just a gain, but caused index errors previously
    synapse = nengo.LinearFilter([3], [2])
    assert allclose(synapse.filt([2.]), 3)

    # differentiator should work properly
    diff = nengo.LinearFilter([1, -1], [1, 0], analog=False)
    assert allclose(diff.filt([1., -1., 2.], y0=0), [1., -2., 3.])

    # Filtering an integer array should cast to a float
    x = np.arange(10, dtype=nengo.rc.int_dtype)
    synapse = nengo.LinearFilter([1], [0.005, 1])
    assert synapse.filt(x).dtype == nengo.rc.float_dtype

    # Throw an error if non-float dtype
    shape = (1,)
    with pytest.raises(ValidationError, match="Only float data types"):
        synapse.make_state(shape, shape, dt=0.001, dtype=np.int32)
    with pytest.raises(ValidationError, match="Only float data types"):
        synapse.make_state(shape, shape, dt=0.001, dtype=np.complex64)


def test_step_errors():
    # error for A.shape[0] != B.shape[0]
    A = np.ones((2, 2))
    B = np.ones((1, 1))
    C = np.ones((1, 2))
    D = np.ones((1, 1))
    X = np.ones((2, 10))
    with pytest.raises(ValidationError, match="Matrices do not meet"):
        LinearFilter.General(A, B, C, D, X)


def test_filt(plt, rng, allclose):
    dt = 1e-3
    tend = 0.5
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 0.1
    tau_dt = tau / dt

    u = rng.normal(size=nt)

    tk = np.arange(0, 30 * tau_dt)
    k = 1. / tau_dt * np.exp(-tk / tau_dt)
    x = np.convolve(u, k, mode='full')[:nt]

    # support lists as input
    y = Lowpass(tau).filt(list(u), dt=dt, y0=0)

    plt.plot(t, x)
    plt.plot(t, y)

    assert allclose(x, y, atol=1e-3, rtol=1e-2)


def test_filtfilt(plt, rng, allclose):
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)
    synapse = Lowpass(0.03)

    u = rng.normal(size=nt)
    x = synapse.filt(u, dt=dt)
    x = synapse.filt(x[::-1], y0=x[-1], dt=dt)[::-1]
    y = synapse.filtfilt(u, dt=dt)

    plt.plot(t, x)
    plt.plot(t, y, '--')

    assert allclose(x, y)


def test_lti_lowpass(rng, plt, allclose):
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

    assert allclose(x, y)


def test_linearfilter_combine(rng, allclose):
    nt = 3000
    tau0, tau1 = 0.01, 0.02
    u = rng.normal(size=(nt, 10))
    x = LinearFilter([1], [tau0 * tau1, tau0 + tau1, 1]).filt(u, y0=0)
    y = Lowpass(tau0).combine(Lowpass(tau1)).filt(u, y0=0)
    assert allclose(x, y)

    with pytest.raises(ValidationError, match="other LinearFilters"):
        Lowpass(0.1).combine(Triangle(0.01))

    with pytest.raises(ValidationError, match="analog and digital"):
        Lowpass(0.1).combine(LinearFilter([1], [1], analog=False))


def test_combined_delay(Simulator, allclose):
    # ensure that two sequential filters has the same output
    # as combining their individual discrete transfer functions
    nt = 50
    tau = 0.005
    dt = 0.001

    sys1 = nengo.Lowpass(tau)
    (num,), den, _ = cont2discrete((sys1.num, sys1.den), dt=dt)
    sys2 = nengo.LinearFilter(
        np.poly1d(num)**2, np.poly1d(den)**2, analog=False)

    with nengo.Network() as model:
        u = nengo.Node(1)
        x = nengo.Node(size_in=1)
        nengo.Connection(u, x, synapse=sys1)
        p1 = nengo.Probe(x, synapse=sys1)
        p2 = nengo.Probe(u, synapse=sys2)

    with Simulator(model, dt=dt) as sim:
        sim.run_steps(nt)

    assert allclose(sim.data[p1], sim.data[p2])

    # Both have two time-step delays:
    # for sys1, this comes from two levels of filtering
    # for sys2, this comes from one level of filtering + delay in sys2
    assert allclose(sim.data[p1][:2], 0)
    assert not allclose(sim.data[p1][2], 0)


def test_synapseparam():
    """SynapseParam must be a Synapse, and converts numbers to LowPass."""
    class Test:
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


def test_argreprs():

    def check_init_args(cls, args):
        assert getfullargspec(cls.__init__).args[1:] == args

    def check_repr(obj):
        assert eval(repr(obj)) == obj

    check_init_args(LinearFilter, ['num', 'den', 'analog', 'method'])
    check_repr(LinearFilter([1, 2], [3, 4]))
    check_repr(LinearFilter([1, 2], [3, 4], analog=False))

    check_init_args(Lowpass, ['tau'])
    check_repr(Lowpass(0.3))

    check_init_args(Alpha, ['tau'])
    check_repr(Alpha(0.3))

    check_init_args(Triangle, ['t'])
    check_repr(Triangle(0.3))


def test_synapse_subclass(Simulator):
    class MySynapse(Synapse):
        pass

    with nengo.Network() as net:
        node_a = nengo.Node([0])
        node_b = nengo.Node(size_in=1)
        nengo.Connection(node_a, node_b, synapse=MySynapse())

    with pytest.raises(NotImplementedError, match="must implement make_state"):
        with Simulator(net):
            pass
