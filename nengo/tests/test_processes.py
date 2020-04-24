import sys

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.base import Process
from nengo.dists import Distribution, Gaussian
from nengo.exceptions import ValidationError
from nengo.linear_system import LinearSystem
from nengo.processes import (
    BrownNoise,
    FilteredNoise,
    Piecewise,
    PresentInput,
    WhiteNoise,
    WhiteSignal,
)
from nengo.synapses import Lowpass


class DistributionMock(Distribution):
    def __init__(self, retval):
        super().__init__()
        self.retval = retval
        self.sample_calls = []

    def sample(self, n, d=None, rng=np.random):
        self.sample_calls.append((n, d, rng))
        return np.ones((n, d)) * self.retval


class TimeProcess(Process):
    def make_step(self, shape_in, shape_out, dt, rng, state):
        size_in = np.prod(shape_in)
        size_out = np.prod(shape_out)
        if size_in == 0:
            return lambda t: [t] * size_out
        return lambda t, x: [t * np.sum(x)] * size_out


class QueueProcess(Process):
    def __init__(self):
        super().__init__()
        self.queue = []

    def make_step(self, shape_in, shape_out, dt, rng, state):
        def step(t, x, queue=self.queue):  # pylint: disable=dangerous-default-value
            queue.append(x)

        return step


def test_time(Simulator, allclose):
    t_run = 1.0
    c = 2.0
    process = TimeProcess()

    with nengo.Network() as model:
        u = nengo.Node(process)
        up = nengo.Probe(u)

        q = nengo.Node(c)
        v = nengo.Node(process, size_in=1, size_out=1)
        nengo.Connection(q, v, synapse=None)
        vp = nengo.Probe(v)

    with Simulator(model) as sim:
        sim.run(t_run)

    nt = len(sim.trange())
    assert process.trange(t_run).shape == sim.trange().shape
    assert allclose(process.trange(t_run), sim.trange())
    assert allclose(process.ntrange(nt), sim.trange())
    assert allclose(process.run(t_run), sim.data[up])
    assert allclose(process.apply([c] * nt), sim.data[vp])


def test_whitenoise(rng):
    dist = DistributionMock(42)
    process = WhiteNoise(dist, scale=False)
    samples = process.run_steps(5, d=2, rng=rng)
    assert np.all(samples == 42 * np.ones((5, 2)))
    assert process.run_steps(5, rng=rng).shape == (5, 1)
    assert process.run_steps(1, d=1, rng=rng).shape == (1, 1)
    assert process.run_steps(2, d=3, rng=rng).shape == (2, 3)


def test_brownnoise(Simulator, seed, plt):
    d = 5000
    t = 0.5
    std = 1.5
    process = BrownNoise(dist=Gaussian(0, std))
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t)
    samples = sim.data[up]

    trange = sim.trange()
    expected_std = std * np.sqrt(trange)
    atol = 3.5 * expected_std / np.sqrt(d)

    plt.subplot(2, 1, 1)
    plt.title("Five Brown noise signals")
    plt.plot(trange, samples[:, :5])
    plt.subplot(2, 1, 2)
    plt.ylabel("Standard deviation")
    plt.plot(trange, np.abs(np.std(samples, axis=1)), label="Actual")
    plt.plot(trange, expected_std, label="Expected")
    plt.legend(loc="best")

    assert np.all(np.abs(np.mean(samples, axis=1)) < atol)
    assert np.all(np.abs(np.std(samples, axis=1) - expected_std) < atol)


def psd(values, dt=0.001):
    freq = npext.rfftfreq(values.shape[0], d=dt)
    power = (
        2.0
        * np.std(np.abs(np.fft.rfft(values, axis=0)), axis=1)
        / np.sqrt(values.shape[0])
    )
    return freq, power


@pytest.mark.parametrize("rms", [0.5, 1, 100])
def test_gaussian_whitenoise(Simulator, rms, seed, plt, allclose):
    d = 800
    process = WhiteNoise(Gaussian(0.0, rms), scale=False)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(0.3)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title(f"First two dimensions of white noise process, rms={rms:.1f}")
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle="steps")

    val_rms = npext.rms(values, axis=0)
    assert allclose(val_rms.mean(), rms, rtol=0.02)
    assert allclose(val_psd[1:-1], rms, rtol=0.2)


@pytest.mark.parametrize("rms", [0.5, 1, 100])
def test_whitesignal_rms(Simulator, rms, seed, plt, allclose):
    t = 1.0
    d = 500
    process = WhiteSignal(t, high=500, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=sim.dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title(f"First two D of white noise process, rms={rms:.1f}")
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle="steps")

    assert allclose(np.std(values), rms, rtol=0.02)
    assert allclose(val_psd[1:-1], rms, rtol=0.35)


@pytest.mark.parametrize("y0,d", [(0, 1), (-0.3, 3), (0.4, 1)])
def test_whitesignal_y0(Simulator, seed, y0, d):
    t = 0.1
    process = WhiteSignal(t, high=500, y0=y0)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t)
    values = sim.data[up]
    error = np.min(abs(y0 - values), axis=0)

    assert ((y0 - error <= values[0, :]) & (values[0, :] <= y0 + error)).all()


@pytest.mark.parametrize("high,dt", [(10, 0.01), (5, 0.001), (50, 0.001)])
def test_whitesignal_high_dt(Simulator, high, dt, seed, plt, allclose):
    t = 1.0
    rms = 0.5
    d = 500
    process = WhiteSignal(t, high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=d)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed, dt=dt) as sim:
        sim.run(t)
    values = sim.data[up]
    freq, val_psd = psd(values, dt=dt)

    trange = sim.trange()
    plt.subplot(2, 1, 1)
    plt.title(f"First two D of white noise process, high={high} Hz")
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle="steps")
    plt.xlim(0, high * 2.0)

    assert allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(len(trange), dt) > high] < rms * 0.5)


@pytest.mark.parametrize("high,dt", [(501, 0.001), (500, 0.002)])
def test_whitesignal_high_errors(Simulator, dt, high, seed):
    """Check for errors if ``high`` is not between 1/period and nyquist frequency."""

    with pytest.raises(ValidationError, match="Make ``high >= 1. / period``"):
        process = WhiteSignal(period=10 * dt, high=9 * dt)

    process = WhiteSignal(1.0, high=high)
    with nengo.Network() as model:
        nengo.Node(process, size_out=1)

    with pytest.raises(ValidationError, match="High must not exceed the Nyquist"):
        Simulator(model, dt=dt, seed=seed)


def test_whitesignal_continuity(Simulator, seed, plt):
    """Test that WhiteSignal is continuous over multiple periods."""
    t = 1.0
    high = 10
    rms = 0.5
    process = WhiteSignal(t, high=high, rms=rms)
    with nengo.Network() as model:
        u = nengo.Node(process, size_out=1)
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(4 * t)
    dt = sim.dt
    x = sim.data[up]

    plt.plot(sim.trange(), x)

    # tolerances approximated from derivatives of sine wave of highest freq
    safety_factor = 2.0
    a, f = np.sqrt(2) * rms, (2 * np.pi * high) * dt
    assert abs(np.diff(x, axis=0)).max() <= safety_factor * a * f
    assert abs(np.diff(x, n=2, axis=0)).max() <= safety_factor ** 2 * a * f ** 2


def test_sampling_shape():
    process = WhiteSignal(0.1, high=500)
    assert process.run_steps(1).shape == (1, 1)
    assert process.run_steps(5, d=1).shape == (5, 1)
    assert process.run_steps(1, d=2).shape == (1, 2)


def test_x_copy(Simulator, allclose):
    """Test that process `x` is copied internally.

    If it is not copied, all elements in `process.queue` will reference the same
    underlying array, and will all be equal to each other.
    """
    with nengo.Network() as model:
        u = nengo.Node(lambda t: [t, t + 2])
        process = QueueProcess()
        v = nengo.Node(process, size_in=2)
        nengo.Connection(u, v, synapse=None)

    with Simulator(model) as sim:
        sim.run(0.003)

    t = sim.trange()
    assert allclose(process.queue, np.column_stack([t, t + 2]))


def test_reset(Simulator, seed, allclose):
    t_run = 0.1

    with nengo.Network() as model:
        u = nengo.Node(WhiteNoise(Gaussian(0, 1), scale=False))
        up = nengo.Probe(u)

    with Simulator(model, seed=seed) as sim:
        sim.run(t_run)
        x = np.array(sim.data[up])
        sim.reset()
        sim.run(t_run)
        y = np.array(sim.data[up])

    assert x.shape == y.shape
    assert allclose(x, y)


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = WhiteNoise(dist=Gaussian(0.3, 0.2))
    b = WhiteNoise(dist=Gaussian(0.3, 0.2))
    c = FilteredNoise(dist=Gaussian(0.3, 0.2), synapse=Lowpass(0.02))

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely

    with pytest.raises(ValueError):
        a.dist = Gaussian(0.3, 0.5)  # test that dist param is frozen
    with pytest.raises(ValueError):
        a.dist.std = 0.4  # test that dist object is frozen


def test_seed(Simulator, seed, allclose):
    with nengo.Network() as model:
        a = nengo.Node(WhiteSignal(0.1, high=100, seed=seed))
        b = nengo.Node(WhiteSignal(0.1, high=100, seed=seed + 1))
        c = nengo.Node(WhiteSignal(0.1, high=100))
        d = nengo.Node(WhiteNoise(seed=seed))
        e = nengo.Node(WhiteNoise())
        ap = nengo.Probe(a)
        bp = nengo.Probe(b)
        cp = nengo.Probe(c)
        dp = nengo.Probe(d)
        ep = nengo.Probe(e)

    with Simulator(model) as sim1:
        sim1.run(0.1)

    with Simulator(model) as sim2:
        sim2.run(0.1)

    tols = dict(atol=1e-7, rtol=1e-4)
    assert allclose(sim1.data[ap], sim2.data[ap], **tols)
    assert allclose(sim1.data[bp], sim2.data[bp], **tols)
    assert not allclose(
        sim1.data[cp], sim2.data[cp], record_rmse=False, print_fail=0, **tols
    )
    assert not allclose(
        sim1.data[ap], sim1.data[bp], record_rmse=False, print_fail=0, **tols
    )
    assert allclose(sim1.data[dp], sim2.data[dp], **tols)
    assert not allclose(
        sim1.data[ep], sim2.data[ep], record_rmse=False, print_fail=0, **tols
    )


def test_present_input(Simulator, rng, allclose):
    n = 5
    c, ni, nj = 3, 8, 8
    images = rng.normal(size=(n, c, ni, nj))
    pres_time = 0.1

    model = nengo.Network()
    with model:
        u = nengo.Node(PresentInput(images, pres_time))
        up = nengo.Probe(u)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    i = (np.floor((t - sim.dt) / pres_time + 1e-7) % n).astype(int)
    y = sim.data[up].reshape(len(t), c, ni, nj)
    for k, [ii, image] in enumerate(zip(i, y)):
        assert allclose(image, images[ii], rtol=1e-4, atol=1e-7), (k, ii)


class TestPiecewise:
    def run_sim(self, data, interpolation, Simulator):
        process = Piecewise(data, interpolation=interpolation)

        with nengo.Network() as model:
            u = nengo.Node(process, size_out=process.default_size_out)
            up = nengo.Probe(u)

        with Simulator(model) as sim:
            sim.run(0.15)

        return sim.trange(), sim.data[up]

    def test_basic(self, Simulator, allclose):
        t, f = self.run_sim({0.05: 1, 0.1: 0}, "zero", Simulator)
        assert allclose(f[t == 0.001], [0.0])
        assert allclose(f[t == 0.025], [0.0])
        assert allclose(f[t == 0.049], [0.0])
        assert allclose(f[t == 0.05], [1.0])
        assert allclose(f[t == 0.075], [1.0])
        assert allclose(f[t == 0.1], [0.0])
        assert allclose(f[t == 0.15], [0.0])

    def test_lists_in_data(self, Simulator, allclose):
        t, f = self.run_sim({0.05: [1, 0], 0.1: [0, 1]}, "zero", Simulator)
        assert allclose(f[t == 0.001], [0.0, 0.0])
        assert allclose(f[t == 0.025], [0.0, 0.0])
        assert allclose(f[t == 0.049], [0.0, 0.0])
        assert allclose(f[t == 0.05], [1.0, 0.0])
        assert allclose(f[t == 0.075], [1.0, 0.0])
        assert allclose(f[t == 0.1], [0.0, 1.0])
        assert allclose(f[t == 0.15], [0.0, 1.0])

    def test_default_zero(self, allclose):
        process = Piecewise({0.05: 1, 0.1: 0})
        f = process.make_step(
            shape_in=(process.default_size_in,),
            shape_out=(process.default_size_out,),
            dt=process.default_dt,
            rng=None,
            state={},
        )
        assert allclose(f(-10), [0.0])
        assert allclose(f(0), [0.0])

    def test_invalid_key(self):
        data = {0.05: 1, 0.1: 0, "a": 0.2}
        with pytest.raises(ValidationError, match=r"Keys must be times \(floats or in"):
            Piecewise(data)

    def test_invalid_callable(self):
        def badcallable(t):
            raise RuntimeError()

        data = {0.05: 1, 0.1: badcallable}
        with pytest.raises(ValidationError, match="should return a numerical const"):
            Piecewise(data)

    def test_invalid_length(self):
        data = {0.05: [1, 0], 0.1: [1, 0, 0]}
        with pytest.raises(ValidationError):
            Piecewise(data)

        # check that scalars and length 1 arrays can be used interchangeably
        # (no validation error)
        Piecewise({0.05: [1], 0.1: 0})

    def test_invalid_interpolation_type(self):
        data = {0.05: 1, 0.1: 0}
        with pytest.raises(ValidationError):
            Piecewise(data, interpolation="not-interpolation")

    def test_fallback_to_zero(self, monkeypatch):
        # Emulate not having scipy in case we have scipy
        monkeypatch.setitem(sys.modules, "scipy.interpolate", None)

        with pytest.warns(UserWarning, match="cannot be applied.*scipy is not inst"):
            process = Piecewise({0.05: 1, 0.1: 0}, interpolation="linear")
        assert process.interpolation == "zero"

    def test_interpolation_1d(self, plt, Simulator, allclose):
        pytest.importorskip("scipy")

        # Note: cubic requires an explicit start of 0
        data = {0: -0.5, 0.05: 1, 0.075: -1, 0.1: -0.5}

        def test_and_plot(interp):
            t, f = self.run_sim(data, interp, Simulator)
            assert allclose(f[t == 0.05], [1.0])
            assert allclose(f[t == 0.075], [-1.0])
            assert allclose(f[t == 0.1], [-0.5])
            plt.plot(t, f, label=interp)

        test_and_plot("zero")
        test_and_plot("linear")
        test_and_plot("nearest")
        test_and_plot("slinear")
        test_and_plot("quadratic")
        test_and_plot("cubic")
        plt.legend(loc="lower left")

    def test_interpolation_2d(self, plt, Simulator, allclose):
        pytest.importorskip("scipy")

        # Note: cubic requires an explicit start of 0
        data = {
            0: [-0.5, -0.5],
            0.05: [1.0, 0.5],
            0.075: [-1, -0.5],
            0.1: [-0.5, -0.25],
        }

        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        def test_and_plot(interp):
            t, f = self.run_sim(data, interp, Simulator)
            assert allclose(f[t == 0.05], [1.0, 0.5])
            assert allclose(f[t == 0.075], [-1.0, -0.5])
            assert allclose(f[t == 0.1], [-0.5, -0.25])
            ax1.plot(t, f.T[0], label=interp)
            ax2.plot(t, f.T[1], label=interp)

        test_and_plot("zero")
        test_and_plot("linear")
        test_and_plot("nearest")
        test_and_plot("slinear")
        test_and_plot("quadratic")
        test_and_plot("cubic")
        plt.legend(loc="lower left")

    @pytest.mark.parametrize("value", (1, [1, 1]))
    def test_shape_out(self, value):
        process = Piecewise({1: value})
        f = process.make_step(
            shape_in=(process.default_size_in,),
            shape_out=(process.default_size_out,),
            dt=process.default_dt,
            rng=None,
            state={},
        )

        assert np.array_equal(f(0), np.zeros(process.default_size_out))
        assert np.array_equal(f(2), np.ones(process.default_size_out))

    def test_function(self, Simulator, allclose):
        t, f = self.run_sim({0.05: np.sin, 0.1: np.cos}, "zero", Simulator)
        assert allclose(f[t == 0.001], [0.0])
        assert allclose(f[t == 0.049], [0.0])
        assert allclose(f[t == 0.05], [np.sin(0.05)])
        assert allclose(f[t == 0.075], [np.sin(0.075)])
        assert allclose(f[t == 0.1], [np.cos(0.1)])
        assert allclose(f[t == 0.15], [np.cos(0.15)])

    def test_function_list(self, Simulator, allclose):
        def func1(t):
            return t, t ** 2, t ** 3

        def func2(t):
            return t ** 4, t ** 5, t ** 6

        t, f = self.run_sim({0.05: func1, 0.1: func2}, "zero", Simulator)
        assert allclose(f[t == 0.001], [0.0])
        assert allclose(f[t == 0.049], [0.0])
        assert allclose(f[t == 0.05], func1(0.05))
        assert allclose(f[t == 0.075], func1(0.075))
        assert allclose(f[t == 0.1], func2(0.1))
        assert allclose(f[t == 0.15], func2(0.15))

    def test_mixture(self, Simulator, allclose):
        t, f = self.run_sim({0.05: 1, 0.1: np.cos}, "zero", Simulator)
        assert allclose(f[t == 0.001], [0.0])
        assert allclose(f[t == 0.049], [0.0])
        assert allclose(f[t == 0.05], [1.0])
        assert allclose(f[t == 0.075], [1.0])
        assert allclose(f[t == 0.1], [np.cos(0.1)])
        assert allclose(f[t == 0.15], [np.cos(0.15)])

    def test_mixture_3d(self, Simulator, allclose):
        def func(t):
            return t, t ** 2, t ** 3

        t, f = self.run_sim({0.05: [1, 1, 1], 0.1: func}, "zero", Simulator)
        assert allclose(f[t == 0.001], [0.0, 0.0, 0.0])
        assert allclose(f[t == 0.049], [0.0, 0.0, 0.0])
        assert allclose(f[t == 0.05], [1.0, 1.0, 1.0])
        assert allclose(f[t == 0.075], [1.0, 1.0, 1.0])
        assert allclose(f[t == 0.1], func(0.1))
        assert allclose(f[t == 0.15], func(0.15))

    def test_invalid_function_length(self):
        with pytest.raises(ValidationError, match="time 1.0 has size 2"):
            Piecewise({0.5: 0, 1.0: lambda t: [t, t ** 2]})

    def test_invalid_interpolation_on_func(self):
        def func(t):
            return t

        with pytest.warns(UserWarning, match="cannot be applied.*callable was sup"):
            process = Piecewise({0.05: 0, 0.1: func}, interpolation="linear")
        assert process.interpolation == "zero"

    def test_cubic_interpolation_warning(self):
        pytest.importorskip("scipy")

        # cubic interpolation with 0 not in times
        process = Piecewise({0.001: 0, 0.1: 0.1}, interpolation="cubic")
        with pytest.warns(UserWarning, match="'cubic' interpolation.*for t=0.0"):
            try:
                process.run(0.001)
            except ValueError:
                pass  # scipy may raise a ValueError


class TestLinearSystem:
    def run_sim(
        self,
        Simulator,
        sys,
        analog=True,
        x0=0,
        dt=0.001,
        simtime=0.3,
        plt=None,
        f_in=None,
    ):
        process = (
            sys
            if isinstance(sys, LinearSystem)
            else LinearSystem(sys, analog=analog, x0=x0)
        )

        with nengo.Network() as model:
            y = nengo.Node(process)
            yp = nengo.Probe(y)

            if f_in is not None:
                u = nengo.Node(f_in)
                nengo.Connection(u, y, synapse=None)

        with Simulator(model, dt=dt, progress_bar=False) as sim:
            sim.run(simtime)

        if plt:
            plt.plot(sim.trange(), sim.data[yp])

        return sim.trange(), sim.data[yp]

    def test_passthrough(self, Simulator, plt, allclose):
        f_in = lambda t: np.array([0.3, 0.7]) * np.sin(5 * t)
        dt = 0.001
        t, y = self.run_sim(
            Simulator, (None, None, None, [[0, 2], [3, 0]]), dt=dt, plt=plt, f_in=f_in
        )
        u = f_in(t[:, None])
        assert allclose(y[:, 0], 2 * u[:, 1], atol=1e-4)
        assert allclose(y[:, 1], 3 * u[:, 0], atol=1e-4)

    def test_autonomous_oscillator(self, Simulator, plt, allclose):
        omega = 6 * np.pi
        A = [[0, -omega], [omega, 0]]
        C = np.eye(2)
        x0 = [1, 0]

        dt = 0.001
        t, y = self.run_sim(
            Simulator, (A, None, C, None), plt=plt, x0=x0, dt=dt, simtime=1.0
        )
        t -= dt
        assert allclose(y[:, 0], np.cos(omega * t), atol=1e-4)
        assert allclose(y[:, 1], np.sin(omega * t), atol=1e-4)

    @pytest.mark.parametrize("dims", [1, 3])
    def test_integrator(self, dims, Simulator, rng, plt, allclose):
        dt = 0.001
        A = None
        B = np.eye(dims)
        C = np.eye(dims)

        freqs = rng.uniform(5 * np.pi, 7 * np.pi, size=dims)
        amps = rng.uniform(0.5, 1.5, size=dims)
        f_in = lambda t: amps * np.sin(freqs * t)

        t, y = self.run_sim(Simulator, (A, B, C, None), dt=dt, plt=plt, f_in=f_in)
        t -= dt
        u = f_in(t[:, None])
        assert allclose(y, np.cumsum(u, axis=0) * dt, atol=1e-4)

    def test_discretize(self, Simulator, plt):
        dt = 0.003
        sys = LinearSystem(([1], [0.02, 1]), method="euler")
        f_in = lambda t: np.sin(20 * t)
        _, y0 = self.run_sim(Simulator, sys, dt=dt, plt=plt, f_in=f_in)
        _, y1 = self.run_sim(Simulator, sys.discretize(dt), dt=dt, plt=plt, f_in=f_in)
        _, y2 = self.run_sim(
            Simulator, sys.discrete_ss(dt), analog=False, dt=dt, plt=plt, f_in=f_in
        )
        assert np.allclose(y1, y0)
        assert np.allclose(y2, y0)

    @pytest.mark.parametrize("A", [0, 0.3])
    @pytest.mark.parametrize("B", [0, 1.3])
    @pytest.mark.parametrize("C", [0, 1.5])
    @pytest.mark.parametrize("D", [0, -0.7])
    @pytest.mark.parametrize("input_size", [0, 2])
    @pytest.mark.parametrize("shape", [(), (3,)])
    def test_all_systems(self, A, B, C, D, input_size, shape, rng):
        state_size = rng.randint(1, 5)
        output_size = rng.randint(1, 4)
        A = A * np.ones((state_size, state_size))
        B = B * np.ones((state_size, input_size))
        C = C * np.ones((output_size, state_size))
        D = D * np.ones((output_size, input_size))

        dt = 0.001
        shape_in = (B.shape[1],) + shape
        shape_out = (C.shape[0],) + shape
        shape_state = (A.shape[0],) + shape
        sys = LinearSystem((A, B, C, D), analog=False)
        u = rng.uniform(-1, 1, size=shape_in)
        x0 = rng.uniform(-1, 1, size=shape_state)

        state = sys.make_state(shape_in, shape_out, dt, x0=x0)
        assert state["X"].shape == shape_state
        assert np.allclose(state["X"], x0)

        step = sys.make_step(shape_in, shape_out, dt, rng=None, state=state)
        if input_size > 0:
            y = step(t=0, u=u)
        else:
            y = step(t=0)  # pylint: disable=no-value-for-parameter
        assert np.allclose(state["X"], A.dot(x0) + B.dot(u))
        assert np.allclose(y, C.dot(x0) + D.dot(u))

    @pytest.mark.parametrize("dims", [1, 2])
    def test_combine(self, dims, rng, plt, allclose):
        """Also tests num-den form with multiple outputs (i.e. multiple num rows)"""
        dt = 0.0015
        tau_a = 0.005
        tau_b = 0.008

        mid_scales = rng.uniform(0.5, 1.5, dims)
        out_scales = rng.uniform(0.5, 1.5, dims)
        scales = mid_scales * out_scales

        sys_ref = LinearSystem((scales[:, None], [tau_a * tau_b, tau_a + tau_b, 1]))

        E = np.eye(dims)
        M = np.diag(mid_scales)
        C = np.diag(out_scales)
        sys1 = LinearSystem((-E / tau_a, E / tau_a, C, None)).combine(
            LinearSystem((mid_scales[:, None], [tau_b, 1]))
        )
        sys2 = LinearSystem((-E / tau_a, M / tau_a, C, None)).combine(
            LinearSystem((np.ones((dims, 1)), [tau_b, 1]))
        )

        t = dt * np.arange(100)
        noise_r = 0.5
        u = rng.uniform(-noise_r, noise_r, size=(len(t), 1))

        out_ref = sys_ref.apply(u, dt=dt)
        out1 = sys1.apply(u, dt=dt)
        out2 = sys2.apply(u, dt=dt)

        plt.plot(t, out1)
        plt.plot(t, out2)
        plt.plot(t, out_ref, ":")

        assert allclose(out1, out_ref, atol=1e-7)
        assert allclose(out2, out_ref, atol=1e-7)

    def test_combine_errors(self):
        sys = (None, np.ones((1, 1)), np.ones((1, 1)), None)
        sys0 = LinearSystem(sys, x0=np.ones((1, 3)))

        # combine with other type
        proc1 = WhiteNoise()
        with pytest.raises(ValidationError, match="Can only combine with"):
            sys0.combine(proc1)

        # combine analog and digital
        sys1 = LinearSystem(sys, analog=False)
        with pytest.raises(ValidationError, match="Cannot combine analog and digital"):
            sys0.combine(sys1)

        # combine with mismatching input/output size
        sys1 = LinearSystem((None, np.ones((1, 1)), np.ones((2, 1)), None))
        with pytest.raises(ValidationError, match="Input size .* must match output"):
            sys0.combine(sys1)

        # combine with mismatching x0 shapes
        sys1 = LinearSystem(sys, x0=np.ones((1, 4)))
        with pytest.raises(ValidationError, match="Initial state shape"):
            sys0.combine(sys1)

    def test_x0(self, rng, plt, allclose):
        dt = 0.001
        nt = 100
        shape = (5,)
        dims = 2
        tau = 0.003

        E = np.eye(dims)
        sys_a = (None, E, E, None)  # integrator
        sys_b = (-E / tau, E / tau, E, None)  # lowpass filter
        sys_a0 = LinearSystem(sys_a)
        sys_b0 = LinearSystem(sys_b)
        sys_ab0 = sys_b0.combine(sys_a0)

        x0a = rng.uniform(-0.01, 0.01, size=(dims,) + shape)
        x0b = rng.uniform(-0.01, 0.01, size=(dims,) + shape)
        u = rng.uniform(-1, 1, size=(nt, dims) + shape)

        ya_ref0 = np.cumsum(np.vstack([0 * u[:1], u]), axis=0)[:-1] * dt
        # yb_ref0 = np.cumsum(np.vstack([0 * ub[:1], ub]), axis=0)[:-1] * dt

        for i in range(x0a.ndim):
            x0ai, x0bi = x0a, x0b
            for _ in range(i):
                x0ai, x0bi = x0ai[..., 0], x0bi[..., 0]

            ya_ref = x0ai.reshape(x0ai.shape + tuple([1] * i)) + ya_ref0

            # test that we can specify x0 at init
            sys_a1 = LinearSystem(sys_a, x0=x0ai)

            # ya0 = sys_a0.apply(u, dt=dt, x0=x0ai)
            ya1 = sys_a1.apply(u, dt=dt)

            # test that we can combine and x0 is preserved
            sys_ab0 = sys_b0.combine(sys_a1)
            yab0_ref = sys_b0.apply(ya_ref)
            yab0 = sys_ab0.apply(u, dt=dt)

            sys_b1 = LinearSystem(sys_b, x0=x0bi)
            sys_ab1 = sys_b1.combine(sys_a1)
            yab1_ref = sys_b1.apply(ya_ref)
            yab1 = sys_ab1.apply(u, dt=dt)

            if i == 0:
                plt.subplot(311)
                plt.plot(ya_ref.reshape(nt, -1), ":")
                plt.plot(ya1.reshape(nt, -1))

                plt.subplot(312)
                plt.plot(yab0_ref.reshape(nt, -1), ":")
                plt.plot(yab0.reshape(nt, -1))

                plt.subplot(313)
                plt.plot(yab1_ref.reshape(nt, -1), ":")
                plt.plot(yab1.reshape(nt, -1))

            # assert allclose(ya0, ya_ref, atol=1e-4)
            assert allclose(ya1, ya_ref, atol=1e-4)

            # higher tolerances because discretizing filters separately is different
            # from discretizing them together
            assert allclose(yab0, yab0_ref, rtol=1e-2, atol=5e-4)
            assert allclose(yab1, yab1_ref, rtol=1e-2, atol=5e-4)

    def test_ss_shape(self):
        A = np.eye(3)
        B = [1, 2, 3]
        C = [1, 2, 3]
        D = None
        ss = LinearSystem((A, B, C, D))
        assert ss.B.shape == (3, 1) and ss.C.shape == (1, 3)

    def test_validation_errors(self):
        with pytest.raises(ValidationError, match="Must be a tuple in"):
            LinearSystem((0, 0, 0, 0, 0))

        with pytest.raises(ValidationError, match="A: Must be a square matrix"):
            LinearSystem((np.ones(3), None, None, None))

        with pytest.raises(ValidationError, match="B: Must be a"):
            LinearSystem((np.ones((1, 1)), np.ones((2, 1)), None, None))

        with pytest.raises(ValidationError, match="C: Must be a"):
            LinearSystem((np.ones((2, 2)), np.ones((2, 1)), np.ones((3, 1)), None))

        with pytest.raises(ValidationError, match="D: Must be a"):
            LinearSystem((np.ones((2, 2)), np.ones((2, 1)), None, np.ones((3, 2))))

        with pytest.raises(ValidationError, match="x0: First dimension"):
            LinearSystem((np.ones((2, 2)), None, None, None), x0=np.ones(3))

        sys = LinearSystem((np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), None))
        with pytest.raises(ValidationError, match="dtype: Only float data types"):
            sys.make_state((2,), (2,), dt=0.001, dtype=np.int32)

    def test_equivalent_formats(self):
        tau0, tau1 = 0.01, 0.02
        sys0 = LinearSystem(([1], [tau0 * tau1, tau0 + tau1, 1]))
        sys1 = LinearSystem(([], [-1 / tau0, -1 / tau1], [1 / (tau0 * tau1)]))

        assert sys0 == sys1
        for M0, M1 in zip(sys0.tf, sys1.tf):
            assert np.allclose(M0, M1)
        for M0, M1 in zip(sys0.zpk, sys1.zpk):
            assert np.allclose(M0, M1)
        for M0, M1 in zip(sys0.ss, sys1.ss):
            assert np.allclose(M0, M1)
