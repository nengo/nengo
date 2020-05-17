import sys

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.base import Process
from nengo.dists import Distribution, Gaussian
from nengo.exceptions import ValidationError
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
        else:
            return lambda t, x: [t * np.sum(x)] * size_out


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
    plt.title("First two dimensions of white noise process, rms=%.1f" % rms)
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
    plt.title("First two D of white noise process, rms=%.1f" % rms)
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
    plt.title("First two D of white noise process, high=%d Hz" % high)
    plt.plot(trange, values[:, :2])
    plt.xlim(right=trange[-1])
    plt.subplot(2, 1, 2)
    plt.title("Power spectrum")
    plt.plot(freq, val_psd, drawstyle="steps")
    plt.xlim(0, high * 2.0)

    assert allclose(np.std(values, axis=1), rms, rtol=0.15)
    assert np.all(val_psd[npext.rfftfreq(len(trange), dt) > high] < rms * 0.5)


@pytest.mark.parametrize("high,dt", [(501, 0.001), (500, 0.002)])
def test_whitesignal_nyquist(Simulator, dt, high, seed):
    # check that high cannot exceed nyquist frequency
    process = WhiteSignal(1.0, high=high)
    with nengo.Network() as model:
        nengo.Node(process, size_out=1)

    with pytest.raises(ValidationError):
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
    assert not allclose(sim1.data[cp], sim2.data[cp], record_rmse=False, **tols)
    assert not allclose(sim1.data[ap], sim1.data[bp], record_rmse=False, **tols)
    assert allclose(sim1.data[dp], sim2.data[dp], **tols)
    assert not allclose(sim1.data[ep], sim2.data[ep], record_rmse=False, **tols)


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

    def test_lists(self, Simulator, allclose):
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
        with pytest.raises(ValidationError):
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

    def test_fallback_to_zero(self, Simulator, monkeypatch):
        # Emulate not having scipy in case we have scipy
        monkeypatch.setitem(sys.modules, "scipy.interpolate", None)

        with pytest.warns(UserWarning):
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

    def test_invalid_function_length(self, Simulator):
        with pytest.raises(ValidationError):
            Piecewise({0.5: 0, 1.0: lambda t: [t, t ** 2]})

    def test_invalid_interpolation_on_func(self):
        def func(t):
            return t

        with pytest.warns(UserWarning):
            process = Piecewise({0.05: 0, 0.1: func}, interpolation="linear")
        assert process.interpolation == "zero"


def test_argreprs():
    """Test repr() for each process type."""
    assert repr(WhiteNoise()) == "WhiteNoise(Gaussian(mean=0, std=1), scale=True)"
    assert (
        repr(WhiteNoise(scale=False))
        == "WhiteNoise(Gaussian(mean=0, std=1), scale=False)"
    )
    assert (
        repr(FilteredNoise()) == "FilteredNoise(synapse=Lowpass(tau=0.005),"
        " dist=Gaussian(mean=0, std=1), scale=True)"
    )
    assert (
        repr(FilteredNoise(scale=False)) == "FilteredNoise(synapse=Lowpass(tau=0.005),"
        " dist=Gaussian(mean=0, std=1), scale=False)"
    )
    assert repr(BrownNoise()) == "BrownNoise(Gaussian(mean=0, std=1))"
    assert (
        repr(PresentInput((1.2, 3.4), 5))
        == "PresentInput(inputs=array([1.2, 3.4]), presentation_time=5)"
    )
    assert repr(WhiteSignal(1, 2)) == "WhiteSignal(period=1, high=2, rms=0.5)"
    assert (
        repr(WhiteSignal(1.2, 3.4, 5.6, 7.8))
        == "WhiteSignal(period=1.2, high=3.4, rms=5.6)"
    )

    assert (
        repr(Piecewise({1: 0.1, 2: 0.2, 3: 0.3}))
        == "Piecewise(data={1: array([0.1]), 2: array([0.2]), 3: array([0.3])})"
    )


def test_piecewise_repr():
    """Test repr() for piecewise."""
    pytest.importorskip("scipy.optimize")
    for interpolation in ("linear", "nearest", "slinear", "quadratic", "cubic"):
        assert (
            repr(Piecewise({1: 0.1, 2: 0.2, 3: 0.3}, interpolation))
            == "Piecewise(data={1: array([0.1]), 2: array([0.2]), 3: array([0.3])}, "
            "interpolation=%r)" % interpolation
        )
