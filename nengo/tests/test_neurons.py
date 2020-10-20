import logging
import time
from collections import defaultdict

import numpy as np
import pytest

import nengo
from nengo.exceptions import BuildError, SimulationError, ValidationError
from nengo.neurons import (
    LIF,
    AdaptiveLIF,
    AdaptiveLIFRate,
    Direct,
    Izhikevich,
    LIFRate,
    NeuronType,
    NeuronTypeParam,
    PoissonSpiking,
    RectifiedLinear,
    RegularSpiking,
    Sigmoid,
    SpikingRectifiedLinear,
    StochasticSpiking,
    Tanh,
)
from nengo.processes import WhiteSignal
from nengo.solvers import LstsqL2nz
from nengo.utils.ensemble import tuning_curves
from nengo.utils.matplotlib import implot, rasterplot
from nengo.utils.numpy import rms


# --- define a composite neuron type, used in `AnyNeuronType` (see `pytest_nengo.py`)
class SpikingTanh(RegularSpiking):
    def __init__(self, tau_ref=0.0025, amplitude=1.0, initial_state=None):
        super().__init__(
            base_type=Tanh(tau_ref=tau_ref),
            amplitude=amplitude,
            initial_state=initial_state,
        )

    @property
    def tau_ref(self):
        return self.base_type.tau_ref


def test_lif_builtin(rng, allclose):
    """Test that the dynamic model approximately matches the rates."""
    dt = 1e-3
    t_final = 1.0

    N = 10
    lif = LIF()
    gain, bias = lif.gain_bias(rng.uniform(80, 100, size=N), rng.uniform(-1, 1, size=N))

    x = np.arange(-2, 2, 0.1)
    J = gain * x[:, None] + bias

    voltage = np.zeros_like(J)
    reftime = np.zeros_like(J)

    spikes = np.zeros((int(t_final / dt),) + J.shape)
    for spikes_i in spikes:
        lif.step(dt, J, spikes_i, voltage, reftime)

    math_rates = lif.rates(x, gain, bias)
    sim_rates = spikes.mean(0)
    assert allclose(sim_rates, math_rates, atol=1, rtol=0.02)


@pytest.mark.parametrize("dt", (0.001, 0.002))
def test_lif(Simulator, plt, rng, dt, allclose):
    """Test that the dynamic model approximately matches the rates."""
    n = 5000
    x = 0.5
    encoders = np.ones((n, 1))
    max_rates = rng.uniform(low=10, high=200, size=n)
    intercepts = rng.uniform(low=-1, high=1, size=n)

    m = nengo.Network()
    with m:
        ins = nengo.Node(x)
        ens = nengo.Ensemble(
            n,
            dimensions=1,
            neuron_type=LIF(),
            encoders=encoders,
            max_rates=max_rates,
            intercepts=intercepts,
        )
        nengo.Connection(ins, ens.neurons, transform=np.ones((n, 1)), synapse=None)
        spike_probe = nengo.Probe(ens.neurons)
        voltage_probe = nengo.Probe(ens.neurons, "voltage")
        ref_probe = nengo.Probe(ens.neurons, "refractory_time")

    t_final = 1.0
    with Simulator(m, dt=dt) as sim:
        sim.run(t_final)

    i = 3
    plt.subplot(311)
    plt.plot(sim.trange(), sim.data[spike_probe][:, :i])
    plt.subplot(312)
    plt.plot(sim.trange(), sim.data[voltage_probe][:, :i])
    plt.subplot(313)
    plt.plot(sim.trange(), sim.data[ref_probe][:, :i])
    plt.ylim([-dt, ens.neuron_type.tau_ref + dt])

    # check rates against analytic rates
    math_rates = ens.neuron_type.rates(
        x, *ens.neuron_type.gain_bias(max_rates, intercepts)
    )
    spikes = sim.data[spike_probe]
    sim_rates = (spikes > 0).sum(0) / t_final
    logging.info("ME = %f", (sim_rates - math_rates).mean())
    logging.info("RMSE = %f", rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
    assert np.sum(math_rates > 0) > 0.5 * n, "At least 50% of neurons must fire"
    assert allclose(sim_rates, math_rates, atol=1, rtol=0.02)

    # if voltage and ref time are non-constant, the probe is doing something
    assert np.abs(np.diff(sim.data[voltage_probe])).sum() > 1
    assert np.abs(np.diff(sim.data[ref_probe])).sum() > 1

    # compute spike counts after each timestep
    actual_counts = (spikes > 0).cumsum(axis=0)
    expected_counts = np.outer(sim.trange(), math_rates)
    assert (abs(actual_counts - expected_counts) < 1).all()


@pytest.mark.parametrize("min_voltage", [-np.inf, -1, 0])
def test_lif_min_voltage(Simulator, plt, min_voltage, seed):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(lambda t: np.sin(t * 4 * np.pi))
        ens = nengo.Ensemble(
            n_neurons=10, dimensions=1, neuron_type=LIF(min_voltage=min_voltage)
        )
        nengo.Connection(stim, ens, synapse=None)
        p_val = nengo.Probe(ens, synapse=0.01)
        p_voltage = nengo.Probe(ens.neurons, "voltage")

    with Simulator(model) as sim:
        sim.run(0.5)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_val])
    plt.ylabel("Decoded value")
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(), sim.data[p_voltage])
    plt.ylabel("Voltage")

    if min_voltage < -10:
        assert np.min(sim.data[p_voltage]) < -10
    else:
        assert np.min(sim.data[p_voltage]) == min_voltage


def test_lif_zero_tau_ref(Simulator, allclose):
    """If we set tau_ref=0, we should be able to reach firing rate 1/dt."""
    dt = 1e-3
    max_rate = 1.0 / dt
    with nengo.Network() as m:
        ens = nengo.Ensemble(
            1, 1, encoders=[[1]], max_rates=[max_rate], neuron_type=LIF(tau_ref=0)
        )
        nengo.Connection(nengo.Node(output=10), ens)
        p = nengo.Probe(ens.neurons)
    with Simulator(m) as sim:
        sim.run(0.02)
    assert allclose(sim.data[p][1:], max_rate)


def test_alif_rate(Simulator, plt, allclose):
    n = 100
    max_rates = 50 * np.ones(n)
    intercepts = np.linspace(-0.99, 0.99, n)
    encoders = np.ones((n, 1))

    model = nengo.Network()
    with model:
        u = nengo.Node(output=0.5)
        a = nengo.Ensemble(
            n,
            1,
            max_rates=max_rates,
            intercepts=intercepts,
            encoders=encoders,
            neuron_type=AdaptiveLIFRate(),
        )
        nengo.Connection(u, a, synapse=None)
        ap = nengo.Probe(a.neurons)

    with Simulator(model) as sim:
        sim.run(2.0)

    t = sim.trange()
    rates = sim.data[ap]
    _, ref = tuning_curves(a, sim, inputs=0.5)

    ax = plt.subplot(211)
    implot(plt, t, intercepts[::-1], rates.T, ax=ax)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("input")
    ax = plt.subplot(212)
    ax.plot(intercepts, ref[::-1].T, "k--")
    ax.plot(intercepts, rates[[1, 500, 1000, -1], ::-1].T)
    ax.set_xlabel("input")
    ax.set_xlabel("rate")

    # check that initial tuning curve is the same as LIF rates
    assert allclose(rates[1], ref, atol=0.1, rtol=1e-3)

    # check that curves in firing region are monotonically decreasing
    assert np.all(np.diff(rates[1:, intercepts < 0.4], axis=0) < 0)


def test_alif(Simulator, plt):
    """Test ALIF and ALIFRate by comparing them to each other"""

    n = 100
    max_rates = 50 * np.ones(n)
    intercepts = np.linspace(-0.99, 0.99, n)
    encoders = np.ones((n, 1))
    nparams = dict(tau_n=1, inc_n=10e-3)
    eparams = dict(
        n_neurons=n, max_rates=max_rates, intercepts=intercepts, encoders=encoders
    )

    model = nengo.Network()
    with model:
        u = nengo.Node(output=0.5)
        a = nengo.Ensemble(
            neuron_type=AdaptiveLIFRate(**nparams), dimensions=1, **eparams
        )
        b = nengo.Ensemble(neuron_type=AdaptiveLIF(**nparams), dimensions=1, **eparams)
        nengo.Connection(u, a, synapse=0)
        nengo.Connection(u, b, synapse=0)
        ap = nengo.Probe(a.neurons)
        bp = nengo.Probe(b.neurons)

    with Simulator(model) as sim:
        sim.run(2.0)

    t = sim.trange()
    a_rates = sim.data[ap]
    spikes = sim.data[bp]
    b_rates = nengo.Lowpass(0.04).filtfilt(spikes)

    tmask = (t > 0.1) & (t < 1.7)
    rel_rmse = rms(b_rates[tmask] - a_rates[tmask]) / rms(a_rates[tmask])

    ax = plt.subplot(311)
    implot(plt, t, intercepts[::-1], a_rates.T, ax=ax)
    ax.set_ylabel("input")
    ax = plt.subplot(312)
    implot(plt, t, intercepts[::-1], b_rates.T, ax=ax)
    ax.set_ylabel("input")
    ax = plt.subplot(313)
    implot(plt, t, intercepts[::-1], (b_rates - a_rates)[tmask].T, ax=ax)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("input")

    assert rel_rmse < 0.07


def test_izhikevich(Simulator, plt, seed):
    """Smoke test for using Izhikevich neurons.

    Tests that the 6 parameter sets listed in the original paper can be
    simulated in Nengo (but doesn't test any properties of them).
    """
    with nengo.Network() as m:
        u = nengo.Node(output=WhiteSignal(0.6, high=10), size_out=1)

        # Seed the ensembles (not network) so we get the same sort of neurons
        ens_args = {"n_neurons": 4, "dimensions": 1, "seed": seed}
        rs = nengo.Ensemble(neuron_type=Izhikevich(), **ens_args)
        ib = nengo.Ensemble(
            neuron_type=Izhikevich(reset_voltage=-55, reset_recovery=4), **ens_args
        )
        ch = nengo.Ensemble(
            neuron_type=Izhikevich(reset_voltage=-50, reset_recovery=2), **ens_args
        )
        fs = nengo.Ensemble(neuron_type=Izhikevich(tau_recovery=0.1), **ens_args)
        lts = nengo.Ensemble(neuron_type=Izhikevich(coupling=0.25), **ens_args)
        rz = nengo.Ensemble(
            neuron_type=Izhikevich(tau_recovery=0.1, coupling=0.26), **ens_args
        )

        ensembles = (rs, ib, ch, fs, lts, rz)
        out = {}
        spikes = {}
        for ens in ensembles:
            nengo.Connection(u, ens)
            out[ens] = nengo.Probe(ens, synapse=0.05)
            spikes[ens] = nengo.Probe(ens.neurons)
        up = nengo.Probe(u)

    with Simulator(m, seed=seed + 1) as sim:
        sim.run(0.6)
    t = sim.trange()

    def plot(ens, title, ix):
        ax = plt.subplot(len(ensembles), 1, ix)
        plt.title(title)
        plt.plot(t, sim.data[out[ens]], c="k", lw=1.5)
        plt.plot(t, sim.data[up], c="k", ls=":")
        ax = ax.twinx()
        ax.set_yticks(())
        rasterplot(t, sim.data[spikes[ens]], ax=ax)

    plt.figure(figsize=(10, 12))
    plot(rs, "Regular spiking", 1)
    plot(ib, "Intrinsically bursting", 2)
    plot(ch, "Chattering", 3)
    plot(fs, "Fast spiking", 4)
    plot(lts, "Low-threshold spiking", 5)
    plot(rz, "Resonator", 6)


@pytest.mark.parametrize("max_rate,intercept", [(300.0, 0.0), (100.0, 1.1)])
def test_sigmoid_response_curves(Simulator, max_rate, intercept, allclose):
    """Check the sigmoid response curve monotonically increases.

    The sigmoid response curve should work fine:

    - if max rate > rate at inflection point and intercept < 1
    - if max rate < rate at inflection point and intercept > 1
    """
    with nengo.Network() as m:
        e = nengo.Ensemble(
            1, 1, neuron_type=Sigmoid(), max_rates=[max_rate], intercepts=[intercept]
        )

    with Simulator(m) as sim:
        pass
    _, y = nengo.utils.ensemble.response_curves(e, sim)
    assert allclose(np.max(y), max_rate)
    assert np.all(y > 0.0)
    assert np.all(np.diff(y) > 0.0)  # monotonically increasing


def test_tanh_invalid():
    """Ensures validation error is thrown when given a large max rate."""
    tanh = Tanh(tau_ref=0.5)
    with pytest.raises(ValidationError, match="Max rates must be below"):
        tanh.gain_bias(max_rates=np.array([100]), intercepts=np.array([0]))


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_sigmoid_invalid(Simulator):
    """Check that invalid sigmoid ensembles raise an error."""

    sigmoid = Sigmoid(tau_ref=0.5)
    with pytest.raises(ValidationError, match="Max rates must be below"):
        sigmoid.gain_bias(max_rates=np.array([100]), intercepts=np.array([0]))

    for max_rate, intercept in [(300.0, 1.1), (300.0, 1.0), (100.0, 0.9), (100, 1.0)]:
        with nengo.Network() as net:
            nengo.Ensemble(
                1,
                1,
                neuron_type=Sigmoid(),
                max_rates=[max_rate],
                intercepts=[intercept],
            )
        with pytest.raises(BuildError, match="lead to neurons with negative"):
            with Simulator(net):
                pass


@pytest.mark.slow
@pytest.mark.parametrize("base_type", [LIFRate(), RectifiedLinear(), Tanh()])
def test_spiking_types(base_type, seed, plt, allclose):
    spiking_types = {
        RegularSpiking: dict(atol=0.05, rmse_target=0.011),
        PoissonSpiking: dict(atol=0.13, rmse_target=0.024),
        StochasticSpiking: dict(atol=0.10, rmse_target=0.019),
    }

    n_neurons = 1000

    with nengo.Network(seed=seed) as net:
        u = nengo.Node(lambda t: np.sin(2 * np.pi * t))
        a = nengo.Ensemble(n_neurons, 1)
        nengo.Connection(u, a)
        u_p = nengo.Probe(u, synapse=0.005)
        a_p = nengo.Probe(a, synapse=0.005)

    neuron_types = {
        spiking_type: spiking_type(base_type) for spiking_type in spiking_types
    }
    neuron_types[None] = base_type

    delay = 5
    results = defaultdict(dict)
    for neuron_type in neuron_types.values():
        a.neuron_type = neuron_type

        with nengo.Simulator(net, seed=seed + 1) as sim:
            timer = time.time()
            sim.run(1.0)
            timer = time.time() - timer

        results[neuron_type]["u"] = sim.data[u_p]
        results[neuron_type]["x"] = sim.data[a_p]
        plt.plot(sim.trange(), sim.data[a_p], label=f"{neuron_type}: t={timer:.3f}")

    plt.plot(sim.trange()[delay:], sim.data[u_p][:-delay], "k--")
    plt.legend(loc=3)

    for spiking_type, tols in spiking_types.items():
        neuron_type = neuron_types[spiking_type]
        res = results[neuron_type]
        x = res["x"][delay:]
        u = res["u"][:-delay]
        assert allclose(x, u, atol=tols["atol"]), spiking_type

        # check that spike noise is the target amount above the base spiking model noise
        rmse = rms(x - u)
        assert allclose(rmse, tols["rmse_target"], atol=0.003, rtol=0.25), spiking_type


def test_dt_dependence(Simulator, NonDirectNeuronType, plt, seed, allclose):
    """Neurons should not wildly change with different dt."""
    freq = 10 * (2 * np.pi)
    input_signal = lambda t: [np.sin(freq * t), np.cos(freq * t)]

    with nengo.Network(seed=seed) as m:
        m.config[nengo.Ensemble].neuron_type = NonDirectNeuronType()
        u = nengo.Node(input_signal, size_out=2)
        pre = nengo.Ensemble(80, dimensions=2)
        square = nengo.Ensemble(80, dimensions=2)
        nengo.Connection(u, pre)
        nengo.Connection(pre, square, function=lambda x: x ** 2)

        activity_p = nengo.Probe(square.neurons, synapse=0.05, sample_every=0.001)
        out_p = nengo.Probe(square, synapse=0.05, sample_every=0.001)

    activity_data = []
    out_data = []
    dts = (0.0001, 0.001)
    colors = ("b", "g", "r")
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    for c, dt in zip(colors, dts):
        with Simulator(m, dt=dt, seed=seed + 1) as sim:
            sim.run(0.1)
        t = sim.trange(sample_every=0.001)
        activity_data.append(sim.data[activity_p])
        out_data.append(sim.data[out_p])
        ax1.plot(t, sim.data[out_p], c=c)
        # Just plot 5 neurons
        ax2.plot(t, sim.data[activity_p][..., :5], c=c)

    ax1.set_xlim(right=t[-1])
    ax1.set_ylabel("Decoded output")
    ax2.set_xlim(right=t[-1])
    ax2.set_ylabel("Neural activity")

    assert rms(activity_data[0] - activity_data[1]) < ((1.0 / dts[-1]) * 0.01)
    assert allclose(out_data[0], out_data[1], atol=0.05)


@pytest.mark.parametrize(
    "Neuron", [LIF, LIFRate, RectifiedLinear, SpikingRectifiedLinear]
)
def test_amplitude(Simulator, seed, rng, plt, allclose, Neuron):
    amp = 0.1
    neuron0 = Neuron()
    neuron = Neuron(amplitude=amp)

    # check static
    x = np.linspace(-5, 30)
    y = amp * neuron0.rates(x, 1.0, 0.0)
    y2 = neuron.rates(x, 1.0, 0.0)
    assert allclose(y, y2, atol=1e-5)

    # check dynamic
    n = 100
    x = 1.0
    encoders = np.ones((n, 1))
    max_rates = rng.uniform(low=20, high=200, size=n)
    intercepts = rng.uniform(low=-1, high=0.8, size=n)

    with nengo.Network(seed=seed) as model:
        ins = nengo.Node(x)
        ens = nengo.Ensemble(
            n,
            dimensions=1,
            neuron_type=neuron,
            encoders=encoders,
            max_rates=max_rates,
            intercepts=intercepts,
        )
        nengo.Connection(ins, ens, synapse=None)
        spike_probe = nengo.Probe(ens.neurons)

    dt = 0.001
    t_final = 0.5
    with Simulator(model, dt=dt) as sim:
        sim.run(t_final)

    spikes = sim.data[spike_probe]
    r = spikes.sum(axis=0) * (dt / t_final)

    gain, bias = neuron0.gain_bias(max_rates, intercepts)
    r0 = amp * neuron0.rates(x, gain, bias).squeeze(axis=0)

    i = np.argsort(r0)
    plt.plot(r0[i])
    plt.plot(r[i])

    error = rms(r - r0) / rms(r0)
    assert (error < 0.02).all()


def test_reset(Simulator, NonDirectNeuronType, seed, allclose):
    """Make sure resetting actually resets."""
    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = NonDirectNeuronType()
        u = nengo.Node(WhiteSignal(0.3, high=10), size_out=2)
        ens = nengo.Ensemble(60, dimensions=2)
        square = nengo.Ensemble(60, dimensions=2)
        nengo.Connection(u, ens)
        nengo.Connection(
            ens, square, function=lambda x: x ** 2, solver=LstsqL2nz(weights=True)
        )
        square_p = nengo.Probe(square, synapse=0.01)

    with Simulator(m, seed=seed + 1) as sim:
        sim.run(0.1)
        sim.run(0.2)
        t1 = sim.trange()
        square1 = np.array(sim.data[square_p])

        sim.reset()
        sim.run(0.3)

    assert allclose(sim.trange(), t1)
    assert allclose(sim.data[square_p], square1)


def test_neurontypeparam():
    """NeuronTypeParam must be a neuron type."""

    class Test:
        ntp = NeuronTypeParam("ntp", default=None)

    inst = Test()
    inst.ntp = LIF()
    assert isinstance(inst.ntp, LIF)
    with pytest.raises(ValueError):
        inst.ntp = "a"


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = LIF()
    b = LIF()
    c = LIF(tau_rc=0.3)
    d = Izhikevich()

    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(c) == hash(c)
    assert hash(d) == hash(d)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert hash(a) != hash(c)  # not guaranteed, but highly likely
    assert b != c
    assert hash(b) != hash(c)  # not guaranteed, but highly likely
    assert a != d
    assert hash(a) != hash(d)  # not guaranteed, but highly likely

    with pytest.raises(ValueError):
        a.tau_rc = 0.3
    with pytest.raises(ValueError):
        d.coupling = 8


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_direct_mode_nonfinite_value(Simulator):
    with nengo.Network() as model:
        e1 = nengo.Ensemble(10, 1, neuron_type=Direct())
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e1, e2, function=lambda x: 1.0 / x)

    with Simulator(model) as sim:
        with pytest.raises(SimulationError):
            sim.run(0.01)


@pytest.mark.parametrize("generic", (True, False))
def test_gain_bias(rng, NonDirectNeuronType, generic, allclose):
    if NonDirectNeuronType == Sigmoid and generic:
        # the generic method doesn't work with sigmoid neurons (because they're
        # always positive). that's not a failure, because the sigmoid neurons
        # never need to use the generic method normally, so we'll just skip
        # it for this test.
        return

    n = 100
    max_rates = rng.uniform(300, 400, size=n)
    intercepts = rng.uniform(-0.5, 0.5, size=n)
    neuron_type = NonDirectNeuronType()
    tolerance = 0.1 if generic else 1e-8

    if generic:
        gain, bias = NeuronType.gain_bias(neuron_type, max_rates, intercepts)
    else:
        gain, bias = neuron_type.gain_bias(max_rates, intercepts)

    assert allclose(neuron_type.rates(1, gain, bias), max_rates, atol=tolerance)

    if NonDirectNeuronType == Sigmoid:
        threshold = 0.5 / neuron_type.tau_ref
    else:
        threshold = 0

    x = (intercepts - tolerance)[np.newaxis, :]
    assert np.all(neuron_type.rates(x, gain, bias) <= threshold)
    x = (intercepts + tolerance)[np.newaxis, :]
    assert np.all(neuron_type.rates(x, gain, bias) > threshold)

    if generic:
        max_rates0, intercepts0 = NeuronType.max_rates_intercepts(
            neuron_type, gain, bias
        )
    else:
        max_rates0, intercepts0 = neuron_type.max_rates_intercepts(gain, bias)

    assert allclose(max_rates, max_rates0, atol=tolerance)
    assert allclose(intercepts, intercepts0, atol=tolerance)


def test_current(rng, allclose):
    neuron_type = NeuronType()
    n_neurons = 20
    gain = rng.rand(n_neurons)
    bias = rng.rand(n_neurons)

    # 3 samples
    x = rng.rand(3)
    current = neuron_type.current(x, gain, bias)
    assert allclose(current, gain * x.reshape(-1, 1) + bias)
    assert current.shape == (3, n_neurons)

    # 10 samples, different values for each neuron
    x = rng.rand(10, n_neurons)
    current = neuron_type.current(x, gain, bias)
    assert allclose(current, gain * x + bias)
    assert current.shape == (10, n_neurons)

    with pytest.raises(ValidationError):
        # Incorrect second dimension
        x = rng.rand(10, 2)
        current = neuron_type.current(x, gain, bias)

    with pytest.raises(ValidationError):
        # Too many dimensions
        x = rng.rand(10, n_neurons, 1)
        current = neuron_type.current(x, gain, bias)


def test_rates_shaping(rng, NonDirectNeuronType):
    neuron_type = NonDirectNeuronType()
    n_neurons = 20
    gain = rng.rand(n_neurons)
    bias = rng.rand(n_neurons)

    # 3 samples
    x = rng.rand(3)
    rates = neuron_type.rates(x, gain, bias)
    assert rates.shape == (3, n_neurons)

    # 10 samples, different values for each neuron
    x = rng.rand(10, n_neurons)
    rates = neuron_type.rates(x, gain, bias)
    assert rates.shape == (10, n_neurons)

    with pytest.raises(ValidationError):
        # Incorrect second dimension
        x = rng.rand(10, 2)
        rates = neuron_type.rates(x, gain, bias)

    with pytest.raises(ValidationError):
        # Too many dimensions
        x = rng.rand(10, n_neurons, 1)
        rates = neuron_type.rates(x, gain, bias)


@pytest.mark.parametrize("neuron_type", [nengo.LIF, nengo.SpikingRectifiedLinear])
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_initial_state(neuron_type, Simulator, seed, plt, allclose):
    n = 50
    initial_voltage = {
        "uniform": nengo.dists.Uniform(0, 1),
        "choice": nengo.dists.Choice([0.5]),
        "linspace": np.linspace(0.1, 0.9, n),
    }

    with nengo.Network(seed=seed) as net:
        probes = {}
        for key, voltage in initial_voltage.items():
            a = nengo.Ensemble(
                50,
                1,
                neuron_type=neuron_type(initial_state={"voltage": voltage}),
                gain=nengo.dists.Choice([1.0]),
                bias=nengo.dists.Choice([0.0]),
            )
            probes[key] = nengo.Probe(a.neurons, "voltage")

    with Simulator(net) as sim:
        sim.run(0.1)

    for i, (key, probe) in enumerate(probes.items()):
        plt.subplot(len(probes), 1, i + 1)
        plt.plot(sim.trange(), sim.data[probe][:, :15])

    for key, voltage in initial_voltage.items():
        probe = probes[key]
        initial_voltage = sim.data[probe][0]
        if neuron_type is nengo.LIF:  # undo decay
            initial_voltage = initial_voltage * np.exp(sim.dt / a.neuron_type.tau_rc)

        if isinstance(voltage, nengo.dists.Uniform):
            histogram, _ = np.histogram(initial_voltage, bins=10)
            histogram = histogram / histogram.mean()
            assert np.abs(histogram - 1).mean() < 0.5
        else:
            if isinstance(voltage, nengo.dists.Distribution):
                voltage = voltage.sample(n)
            assert allclose(initial_voltage, voltage, atol=1e-5)


def test_bad_initial_state(rng, Simulator):
    with pytest.raises(ValidationError, match="State variable 'rates' not recognized"):
        nengo.LIF(initial_state={"rates": nengo.dists.Choice([0])})

    with pytest.raises(ValidationError, match="State variable 'voltage' must be a"):
        nengo.LIF(initial_state={"voltage": "voltage"})

    class MyLIF(nengo.LIF):
        def __init__(self, bad_state=False):
            super().__init__()
            self.bad_state = bad_state
            if bad_state:
                self.state = {"in": nengo.dists.Choice([0])}

        def make_state(self, n_neurons, rng=np.random, dtype=None):
            if self.bad_state:
                return super().make_state(n_neurons, rng=np.random, dtype=None)
            return {"rng": rng, "dict": {}}

    with nengo.Network() as net:
        nengo.Ensemble(1, 1, neuron_type=MyLIF(bad_state=True))

    with pytest.raises(BuildError, match="State name 'in' overlaps"):
        with Simulator(net):
            pass

    with nengo.Network() as net:
        nengo.Ensemble(1, 1, neuron_type=MyLIF())

    with pytest.raises(BuildError, match="State 'dict' is of type 'dict'. Only"):
        with Simulator(net):
            pass


def test_spikes_to_spikes_warning():
    with pytest.warns(UserWarning, match="type 'LIF', which is a spiking"):
        nengo.PoissonSpiking(nengo.LIF())
