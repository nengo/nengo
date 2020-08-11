import numpy as np
import pytest

import nengo
from nengo.exceptions import ValidationError
from nengo.processes import WhiteSignal
from nengo.synapses import (
    Alpha,
    Bandpass,
    DiscreteDelay,
    DoubleExp,
    Highpass,
    LegendreDelay,
    LinearFilter,
    Lowpass,
    Synapse,
    SynapseParam,
    Triangle,
)
from nengo.utils.numpy import rms
from nengo.utils.testing import signals_allclose

# The following num, den are for a 4th order analog Butterworth filter,
# generated with `scipy.signal.butter(4, 0.1, analog=False)`
butter_num = np.array([0.0004166, 0.0016664, 0.0024996, 0.0016664, 0.0004166])
butter_den = np.array([1.0, -3.18063855, 3.86119435, -2.11215536, 0.43826514])


def run_synapse(
    Simulator,
    seed,
    synapse,
    dt=1e-3,
    runtime=0.2,
    high=100,
    n_neurons=None,
    ax=None,
):
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

    if ax is not None:
        freqs = np.fft.rfftfreq(len(sim.data[ref]), d=dt)
        ref_fft = np.fft.rfft(sim.data[ref], axis=0)
        filtered_fft = np.fft.rfft(sim.data[filtered], axis=0)
        ax.plot(freqs, np.abs(ref_fft), "k")
        ax.plot(freqs, np.abs(filtered_fft))

    return sim.trange(), sim.data[ref], sim.data[filtered]


@pytest.mark.parametrize("form", ["tf", "zpk", "ss"])
def test_direct(form, Simulator, plt, seed, allclose):
    dt = 1e-3
    a = 0.7
    sys = {"tf": ([a], [1]), "zpk": ([], [], a), "ss": ([], [], [], [[a]])}[form]

    synapse = LinearFilter(sys, analog=False)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, allclose=allclose)
    assert signals_allclose(t, a * x, y, plt=plt, allclose=allclose)


def test_lowpass(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.01

    synapse = Lowpass(tau)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = Lowpass(tau).filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, plt=plt, allclose=allclose)

    gains = np.abs(synapse.evaluate([0, synapse.cutoff, 1e16]))
    assert allclose(gains, [1, 1 / np.sqrt(2), 0])


def test_alpha(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.01
    num, den = [1], [tau ** 2, 2 * tau, 1]

    synapse = Alpha(tau)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = LinearFilter((num, den)).filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt, allclose=allclose)

    check_synapse = Lowpass(tau).combine(Lowpass(tau))
    assert np.allclose(check_synapse.tf[0], synapse.tf[0])
    assert np.allclose(check_synapse.tf[1], synapse.tf[1])

    gains = np.abs(synapse.evaluate([0, synapse.cutoff, 1e16]))
    assert allclose(gains, [1, 1 / np.sqrt(2), 0])


def test_double_exp(Simulator, plt, seed, allclose):
    dt = 1e-3
    synapse = DoubleExp(0.01, 0.002)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)
    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt, allclose=allclose)

    check_synapse = Lowpass(0.002).combine(Lowpass(0.01))
    assert np.allclose(check_synapse.tf[0], synapse.tf[0])
    assert np.allclose(check_synapse.tf[1], synapse.tf[1])

    freqs = [0, synapse.cutoff_low, synapse.cutoff_high, 1e16]
    gains = np.abs(synapse.evaluate(freqs))
    assert (np.diff(freqs) > 0.1).all() and (np.diff(gains) < 0.01).all()
    assert allclose(gains[[0, -1]], [1, 0])
    assert allclose(gains[1], 1 / np.sqrt(2), atol=0.05)


def test_bandpass(Simulator, plt, seed, allclose):
    dt = 1e-3
    synapse = Bandpass(freq=25, alpha=0.1)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)
    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt, allclose=allclose)

    freqs = [0, synapse.cutoff_low, synapse.freq, synapse.cutoff_high, 1e16]
    gains = np.abs(synapse.evaluate(freqs))
    assert (np.diff(freqs) > 0.1).all()
    assert allclose(gains, [0, 1 / np.sqrt(2), 1, 1 / np.sqrt(2), 0])


@pytest.mark.parametrize("tau, order", [(0.002, 1), (0.003, 3)])
def test_highpass(tau, order, Simulator, plt, seed, allclose):
    dt = 1e-3
    synapse = Highpass(tau, order=order)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)
    assert signals_allclose(t, y, yhat, delay=dt, atol=5e-6, plt=plt, allclose=allclose)

    freqs = [0, synapse.cutoff, 1e16]
    gains = np.abs(synapse.evaluate(freqs))
    assert (np.diff(freqs) > 0.1).all()
    assert allclose(gains, np.array([0, 1 / np.sqrt(2), 1]) ** order)


def test_triangle(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.03

    t, x, ysim = run_synapse(Simulator, seed, Triangle(tau), dt=dt)
    yfilt = Triangle(tau).filt(x, dt=dt, y0=0)

    # compare with convolved filter
    n_taps = int(round(tau / dt)) + 1
    num = np.arange(n_taps, 0, -1, dtype=nengo.rc.float_dtype)
    num /= num.sum()
    y = np.convolve(x.ravel(), num)[: len(t)]
    y.shape = (-1, 1)

    assert allclose(y, yfilt, rtol=0)
    assert signals_allclose(t, y, ysim, delay=dt, rtol=0, plt=plt, allclose=allclose)

    # test y0 != 0
    assert allclose(Triangle(tau).filt(np.ones(100), dt=dt, y0=1), 1)


def test_decoders(Simulator, plt, seed, allclose):
    dt = 1e-3
    tau = 0.01

    t, x, yhat = run_synapse(Simulator, seed, Lowpass(tau), dt=dt, n_neurons=100)

    y = Lowpass(tau).filt(x, dt=dt, y0=0)
    assert signals_allclose(t, y, yhat, delay=dt, plt=plt, allclose=allclose)


def test_linearfilter(Simulator, plt, seed, allclose):
    dt = 1e-3
    synapse = LinearFilter((butter_num, butter_den), analog=False)
    t, x, yhat = run_synapse(Simulator, seed, synapse, dt=dt)
    y = synapse.filt(x, dt=dt, y0=0)

    assert signals_allclose(t, y, yhat, delay=dt, plt=plt, allclose=allclose)


def test_linearfilter_evaluate(plt):
    tau = 0.02
    ord1 = LinearFilter(([1], [tau, 1]))
    ord2 = LinearFilter(([1], [tau ** 2, 2 * tau, 1]))

    f = np.logspace(-1, 3, 100)
    y1 = ord1.evaluate(f)
    y2 = ord2.evaluate(f)

    plt.subplot(211)
    plt.semilogx(f, 20 * np.log10(np.abs(y1)))
    plt.semilogx(f, 20 * np.log10(np.abs(y2)))

    plt.subplot(212)
    plt.semilogx(f, np.angle(y1))
    plt.semilogx(f, np.angle(y2))

    jw_tau = 2.0j * np.pi * f * tau
    y1_ref = 1 / (jw_tau + 1)
    y2_ref = 1 / (jw_tau ** 2 + 2 * jw_tau + 1)
    assert np.allclose(y1, y1_ref)
    assert np.allclose(y2, y2_ref)


def test_linearfilter_y0(allclose):
    # --- y0 sets initial state correctly for high-order filter
    synapse = LinearFilter((butter_num, butter_den), analog=False)
    v = 9.81
    x = v * np.ones(10)
    assert allclose(synapse.filt(x, y0=v), v)
    assert not allclose(synapse.filt(x, y0=0), v, record_rmse=False, print_fail=0)

    # --- y0 does not work for high-order synapse when DC gain is zero
    synapse = LinearFilter(([1, 0], [1, 1]))
    with pytest.raises(ValidationError, match="Cannot solve for state"):
        synapse.filt(np.ones(10), y0=1)


def test_linearfilter_extras(allclose):
    # This filter is just a gain, but caused index errors previously
    synapse = nengo.LinearFilter(([3], [2]))
    assert allclose(synapse.filt([2.0]), 3)

    # differentiator should work properly
    diff = nengo.LinearFilter(([1, -1], [1, 0]), analog=False)
    assert allclose(diff.filt([1.0, -1.0, 2.0], y0=0), [1.0, -2.0, 3.0])

    # Filtering an integer array should cast to a float
    x = np.arange(10, dtype=nengo.rc.int_dtype)
    synapse = nengo.LinearFilter(([1], [0.005, 1]))
    assert synapse.filt(x).dtype == nengo.rc.float_dtype

    # Throw an error if non-float dtype
    shape = (1,)
    with pytest.raises(ValidationError, match="Only float data types"):
        synapse.make_state(shape, shape, dt=0.001, rng=None, dtype=np.int32)
    with pytest.raises(ValidationError, match="Only float data types"):
        synapse.make_state(shape, shape, dt=0.001, rng=None, dtype=np.complex64)


def test_linearfilter_den():
    sys = ([1], [0.005, 1])
    with pytest.warns(DeprecationWarning, match="`den` is deprecated"):
        synapse = nengo.LinearFilter(*sys)

    assert synapse == nengo.LinearFilter(sys)


@pytest.mark.parametrize(
    "synapse, rmse_range",
    [
        (DiscreteDelay(20), [0, 1e-8]),
        (LegendreDelay(0.09, 9), [0.001, 0.03]),
        (LegendreDelay(0.09, 21), [0.001, 0.03]),
    ],
)
def test_delay_synapses(synapse, rmse_range, Simulator, seed, plt, allclose):
    simtime = 0.5
    dt = 0.001
    if isinstance(synapse, DiscreteDelay):
        # as per DiscreteDelay docstring notes, delay is one less in filt than in sim
        delay_steps = synapse.steps - 1
    else:
        delay = synapse.theta
        delay_steps = int(round(delay / dt))
        assert delay_steps > 0 and np.allclose(delay_steps * dt, delay, atol=1e-5)

    process = nengo.processes.WhiteSignal(10.0, high=15, y0=0, seed=seed, default_dt=dt)

    # out of simulator
    u = process.run(simtime)
    t = process.ntrange(len(u))
    y = synapse.filt(u, dt=dt)

    # in simulator
    with nengo.Network() as net:
        inp = nengo.Node(process)
        out = nengo.Probe(inp, synapse=synapse)
    with Simulator(net, dt=dt) as sim:
        sim.run(simtime)

    plt.plot(t[delay_steps:], u[:-delay_steps], "--")
    plt.plot(t, y)
    plt.plot(t[:-1], sim.data[out][1:])  # remove simulator delay of 1

    rmse_filt = rms(y[delay_steps:] - u[:-delay_steps])
    rmse_sim = rms(sim.data[out][delay_steps + 1 :] - u[: -delay_steps - 1])
    assert rmse_range[0] <= rmse_filt <= rmse_range[1]
    assert np.allclose(rmse_sim, rmse_filt, atol=1e-5, rtol=5e-3)


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
    k = 1.0 / tau_dt * np.exp(-tk / tau_dt)
    x = np.convolve(u, k, mode="full")[:nt]

    # support lists as input
    y = Lowpass(tau).filt(list(u), dt=dt, y0=0)

    plt.plot(t, x)
    plt.plot(t, y)

    assert allclose(x, y, atol=1e-3, rtol=1e-2)


def test_filtfilt(plt, rng, allclose):
    dt = 1e-3
    tend = 3.0
    t = dt * np.arange(tend / dt)
    nt = len(t)
    synapse = Lowpass(0.03)

    u = rng.normal(size=nt)
    x = synapse.filt(u, dt=dt)
    x = synapse.filt(x[::-1], y0=x[-1], dt=dt)[::-1]
    y = synapse.filtfilt(u, dt=dt)

    plt.plot(t, x)
    plt.plot(t, y, "--")

    assert allclose(x, y)


def test_lti_lowpass(rng, plt, allclose):
    dt = 1e-3
    tend = 3.0
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2
    lti = LinearFilter(([1], [tau, 1]))

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
    x = LinearFilter(([], [-1 / tau0, -1 / tau1], 1 / (tau0 * tau1))).filt(u, y0=0)
    y = Lowpass(tau0).combine(Lowpass(tau1)).filt(u, y0=0)
    assert allclose(x, y)

    with pytest.raises(ValidationError, match="other LinearFilter"):
        Lowpass(0.1).combine(Triangle(0.01))

    with pytest.raises(ValidationError, match="analog and digital"):
        Lowpass(0.1).combine(LinearFilter(([1], [1]), analog=False))


def test_linearfilter_validation_errors():
    with pytest.raises(ValidationError, match="can only have one input"):
        LinearFilter((-np.ones((2, 2)), np.ones((2, 2)), None, None))

    with pytest.raises(ValidationError, match="can only have one output"):
        LinearFilter((-np.ones((2, 2)), np.ones((2, 1)), np.ones((2, 2)), None))


def test_combined_delay(Simulator, allclose):
    # ensure that two sequential filters has the same output
    # as combining their individual discrete transfer functions
    nt = 50
    tau = 0.005
    dt = 0.001

    sys1 = nengo.Lowpass(tau)
    sys1d = sys1.discretize(dt)
    sys2d = sys1d.combine(sys1d)

    with nengo.Network() as model:
        u = nengo.Node(1)
        x = nengo.Node(size_in=1)
        nengo.Connection(u, x, synapse=sys1)
        p1 = nengo.Probe(x, synapse=sys1)
        p2 = nengo.Probe(u, synapse=sys2d)

    with Simulator(model, dt=dt) as sim:
        sim.run_steps(nt)

    assert allclose(sim.data[p1], sim.data[p2])

    # Both have two time-step delays:
    # for sys1, this comes from two levels of filtering
    # for sys2, this comes from one level of filtering + delay in sys2
    assert allclose(sim.data[p1][:2], 0)
    assert not allclose(sim.data[p1][2], 0, record_rmse=False, print_fail=0)


def test_synapseparam():
    """SynapseParam must be a Synapse, and converts numbers to Lowpass."""

    class Test:
        sp = SynapseParam("sp", default=Lowpass(0.1))

    inst = Test()
    assert isinstance(inst.sp, Lowpass)
    assert inst.sp.tau == 0.1
    # Numbers are converted to Lowpass
    inst.sp = 0.05
    assert isinstance(inst.sp, Lowpass)
    assert inst.sp.tau == 0.05
    # None has meaning
    inst.sp = None
    assert inst.sp is None
    # Non-synapse not OK
    with pytest.raises(ValueError):
        inst.sp = "a"


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = LinearFilter(([1], [0.04, 1]))
    b = LinearFilter(([1], [0.04, 1]))
    c = LinearFilter(([1], [0.04, 1.1]))

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
        a.A[0, 0] = 9


def test_synapse_subclass(Simulator):
    class MySynapse(Synapse):  # pylint: disable=abstract-method
        pass

    with nengo.Network() as net:
        node_a = nengo.Node([0])
        node_b = nengo.Node(size_in=1)
        nengo.Connection(node_a, node_b, synapse=MySynapse())

    with pytest.raises(NotImplementedError, match="must implement make_state"):
        with Simulator(net):
            pass
