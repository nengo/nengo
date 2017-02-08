import numpy as np
import pytest

import nengo

from nengo.exceptions import BuildError, SimulationError
from nengo.neurons import Direct, NeuronTypeParam
from nengo.processes import WhiteSignal
from nengo.solvers import LstsqL2nz
from nengo.utils.ensemble import tuning_curves
from nengo.utils.matplotlib import implot, rasterplot
from nengo.utils.neurons import rates_kernel
from nengo.utils.numpy import rms, rmse


def test_lif_builtin(rng):
    """Test that the dynamic model approximately matches the rates."""
    dt = 1e-3
    t_final = 1.0

    N = 10
    lif = nengo.LIF()
    gain, bias = lif.gain_bias(
        rng.uniform(80, 100, size=N), rng.uniform(-1, 1, size=N))

    x = np.arange(-2, 2, .1).reshape(-1, 1)
    J = gain * x + bias

    voltage = np.zeros_like(J)
    reftime = np.zeros_like(J)

    spikes = np.zeros((int(t_final / dt),) + J.shape)
    for i, spikes_i in enumerate(spikes):
        lif.step_math(dt, J, spikes_i, voltage, reftime)

    math_rates = lif.rates(x, gain, bias)
    sim_rates = spikes.mean(0)
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)


@pytest.mark.parametrize('dt', (0.001, 0.002))
def test_lif(Simulator, plt, rng, logger, dt):
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
            n, dimensions=1, neuron_type=nengo.LIF(),
            encoders=encoders, max_rates=max_rates, intercepts=intercepts)
        nengo.Connection(
            ins, ens.neurons, transform=np.ones((n, 1)), synapse=None)
        spike_probe = nengo.Probe(ens.neurons)
        voltage_probe = nengo.Probe(ens.neurons, 'voltage')
        ref_probe = nengo.Probe(ens.neurons, 'refractory_time')

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
        x, *ens.neuron_type.gain_bias(max_rates, intercepts))
    spikes = sim.data[spike_probe]
    sim_rates = (spikes > 0).sum(0) / t_final
    logger.info("ME = %f", (sim_rates - math_rates).mean())
    logger.info("RMSE = %f",
                rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
    assert np.sum(math_rates > 0) > 0.5 * n, (
        "At least 50% of neurons must fire")
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)

    # if voltage and ref time are non-constant, the probe is doing something
    assert np.abs(np.diff(sim.data[voltage_probe])).sum() > 1
    assert np.abs(np.diff(sim.data[ref_probe])).sum() > 1

    # compute spike counts after each timestep
    actual_counts = (spikes > 0).cumsum(axis=0)
    expected_counts = np.outer(sim.trange(), math_rates)
    assert (abs(actual_counts - expected_counts) < 1).all()


@pytest.mark.parametrize('min_voltage', [-np.inf, -1, 0])
def test_lif_min_voltage(Simulator, plt, min_voltage, seed):
    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(lambda t: np.sin(t * 4 * np.pi))
        ens = nengo.Ensemble(n_neurons=10, dimensions=1,
                             neuron_type=nengo.LIF(min_voltage=min_voltage))
        nengo.Connection(stim, ens, synapse=None)
        p_val = nengo.Probe(ens, synapse=0.01)
        p_voltage = nengo.Probe(ens.neurons, 'voltage')

    with Simulator(model) as sim:
        sim.run(0.5)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p_val])
    plt.ylabel("Decoded value")
    plt.subplot(2, 1, 2)
    plt.plot(sim.trange(), sim.data[p_voltage])
    plt.ylabel("Voltage")

    if min_voltage < -100:
        assert np.min(sim.data[p_voltage]) < -100
    else:
        assert np.min(sim.data[p_voltage]) == min_voltage


def test_lif_zero_tau_ref(Simulator):
    """If we set tau_ref=0, we should be able to reach firing rate 1/dt."""
    dt = 1e-3
    max_rate = 1. / dt
    with nengo.Network() as m:
        ens = nengo.Ensemble(1, 1,
                             encoders=[[1]],
                             max_rates=[max_rate],
                             neuron_type=nengo.LIF(tau_ref=0))
        nengo.Connection(nengo.Node(output=10), ens)
        p = nengo.Probe(ens.neurons)
    with Simulator(m) as sim:
        sim.run(0.02)
    assert np.allclose(sim.data[p][1:], max_rate)


def test_alif_rate(Simulator, plt):
    n = 100
    max_rates = 50 * np.ones(n)
    # max_rates = 200 * np.ones(n)
    intercepts = np.linspace(-0.99, 0.99, n)
    encoders = np.ones((n, 1))

    model = nengo.Network()
    with model:
        u = nengo.Node(output=0.5)
        a = nengo.Ensemble(n, 1,
                           max_rates=max_rates,
                           intercepts=intercepts,
                           encoders=encoders,
                           neuron_type=nengo.AdaptiveLIFRate())
        nengo.Connection(u, a, synapse=None)
        ap = nengo.Probe(a.neurons)

    dt = 1e-3
    with Simulator(model, dt=dt) as sim:
        sim.run(2.)

    t = sim.trange()
    rates = sim.data[ap]
    _, ref = tuning_curves(a, sim, inputs=np.array([0.5]))

    ax = plt.subplot(211)
    implot(plt, t, intercepts[::-1], rates.T, ax=ax)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('input')
    ax = plt.subplot(212)
    ax.plot(intercepts, ref[::-1].T, 'k--')
    ax.plot(intercepts, rates[[1, 500, 1000, -1], ::-1].T)
    ax.set_xlabel('input')
    ax.set_xlabel('rate')

    # check that initial tuning curve is the same as LIF rates
    assert np.allclose(rates[1], ref, atol=0.1, rtol=1e-3)

    # check that curves in firing region are monotonically decreasing
    assert np.all(np.diff(rates[1:, intercepts < 0.4], axis=0) < 0)


def test_alif(Simulator, plt):
    """Test ALIF and ALIFRate by comparing them to each other"""

    n = 100
    max_rates = 50 * np.ones(n)
    intercepts = np.linspace(-0.99, 0.99, n)
    encoders = np.ones((n, 1))
    nparams = dict(tau_n=1, inc_n=10e-3)
    eparams = dict(n_neurons=n, max_rates=max_rates,
                   intercepts=intercepts, encoders=encoders)

    model = nengo.Network()
    with model:
        u = nengo.Node(output=0.5)
        a = nengo.Ensemble(neuron_type=nengo.AdaptiveLIFRate(**nparams),
                           dimensions=1,
                           **eparams)
        b = nengo.Ensemble(neuron_type=nengo.AdaptiveLIF(**nparams),
                           dimensions=1,
                           **eparams)
        nengo.Connection(u, a, synapse=0)
        nengo.Connection(u, b, synapse=0)
        ap = nengo.Probe(a.neurons)
        bp = nengo.Probe(b.neurons)

    dt = 1e-3
    with Simulator(model, dt=dt) as sim:
        sim.run(2.)

    t = sim.trange()
    a_rates = sim.data[ap]
    spikes = sim.data[bp]
    b_rates = rates_kernel(t, spikes)

    tmask = (t > 0.1) & (t < 1.7)
    rel_rmse = rms(b_rates[tmask] - a_rates[tmask]) / rms(a_rates[tmask])

    ax = plt.subplot(311)
    implot(plt, t, intercepts[::-1], a_rates.T, ax=ax)
    ax.set_ylabel('input')
    ax = plt.subplot(312)
    implot(plt, t, intercepts[::-1], b_rates.T, ax=ax)
    ax.set_ylabel('input')
    ax = plt.subplot(313)
    implot(plt, t, intercepts[::-1], (b_rates - a_rates)[tmask].T, ax=ax)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('input')

    assert rel_rmse < 0.07


def test_izhikevich(Simulator, plt, seed, rng):
    """Smoke test for using Izhikevich neurons.

    Tests that the 6 parameter sets listed in the original paper can be
    simulated in Nengo (but doesn't test any properties of them).
    """
    with nengo.Network() as m:
        u = nengo.Node(output=WhiteSignal(0.6, high=10), size_out=1)

        # Seed the ensembles (not network) so we get the same sort of neurons
        ens_args = {'n_neurons': 4, 'dimensions': 1, 'seed': seed}
        rs = nengo.Ensemble(neuron_type=nengo.Izhikevich(), **ens_args)
        ib = nengo.Ensemble(neuron_type=nengo.Izhikevich(
            reset_voltage=-55, reset_recovery=4), **ens_args)
        ch = nengo.Ensemble(neuron_type=nengo.Izhikevich(
            reset_voltage=-50, reset_recovery=2), **ens_args)
        fs = nengo.Ensemble(neuron_type=nengo.Izhikevich(tau_recovery=0.1),
                            **ens_args)
        lts = nengo.Ensemble(neuron_type=nengo.Izhikevich(coupling=0.25),
                             **ens_args)
        rz = nengo.Ensemble(neuron_type=nengo.Izhikevich(
            tau_recovery=0.1, coupling=0.26), **ens_args)

        ensembles = (rs, ib, ch, fs, lts, rz)
        out = {}
        spikes = {}
        for ens in ensembles:
            nengo.Connection(u, ens)
            out[ens] = nengo.Probe(ens, synapse=0.05)
            spikes[ens] = nengo.Probe(ens.neurons)
        up = nengo.Probe(u)

    with Simulator(m, seed=seed+1) as sim:
        sim.run(0.6)
    t = sim.trange()

    def plot(ens, title, ix):
        ax = plt.subplot(len(ensembles), 1, ix)
        plt.title(title)
        plt.plot(t, sim.data[out[ens]], c='k', lw=1.5)
        plt.plot(t, sim.data[up], c='k', ls=':')
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


@pytest.mark.parametrize("max_rate,intercept", [(300., 0.0), (100., 1.1)])
def test_sigmoid_response_curves(Simulator, max_rate, intercept):
    """Check the sigmoid response curve monotonically increases.

    The sigmoid response curve should work fine:

    - if max rate > rate at inflection point and intercept < 1
    - if max rate < rate at inflection point and intercept > 1
    """
    with nengo.Network() as m:
        e = nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(),
                           max_rates=[max_rate], intercepts=[intercept])

    with Simulator(m) as sim:
        pass
    x, y = nengo.utils.ensemble.response_curves(e, sim)
    assert np.allclose(np.max(y), max_rate)
    assert np.all(y > 0.)
    assert np.all(np.diff(y) > 0.)  # monotonically increasing


@pytest.mark.parametrize("max_rate,intercept", [
    (300., 1.1), (300., 1.0), (100., 0.9), (100, 1.0)])
def test_sigmoid_invalid(Simulator, max_rate, intercept):
    """Check that invalid sigmoid ensembles raise an error."""
    with nengo.Network() as m:
        nengo.Ensemble(1, 1, neuron_type=nengo.Sigmoid(),
                       max_rates=[max_rate], intercepts=[intercept])
    with pytest.raises(BuildError):
        with Simulator(m):
            pass


def test_dt_dependence(Simulator, nl_nodirect, plt, seed, rng):
    """Neurons should not wildly change with different dt."""
    freq = 10 * (2 * np.pi)
    input_signal = lambda t: [np.sin(freq*t), np.cos(freq*t)]

    with nengo.Network(seed=seed) as m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(input_signal, size_out=2)
        pre = nengo.Ensemble(60, dimensions=2)
        square = nengo.Ensemble(60, dimensions=2)
        nengo.Connection(u, pre)
        nengo.Connection(pre, square, function=lambda x: x ** 2)

        activity_p = nengo.Probe(square.neurons, synapse=.05,
                                 sample_every=0.001)
        out_p = nengo.Probe(square, synapse=.05, sample_every=0.001)

    activity_data = []
    out_data = []
    dts = (0.0001, 0.001)
    colors = ('b', 'g', 'r')
    for c, dt in zip(colors, dts):
        with Simulator(m, dt=dt, seed=seed+1) as sim:
            sim.run(0.1)
        t = sim.trange(dt=0.001)
        activity_data.append(sim.data[activity_p])
        out_data.append(sim.data[out_p])
        plt.subplot(2, 1, 1)
        plt.plot(t, sim.data[out_p], c=c)
        plt.subplot(2, 1, 2)
        # Just plot 5 neurons
        plt.plot(t, sim.data[activity_p][..., :5], c=c)

    plt.subplot(2, 1, 1)
    plt.xlim(right=t[-1])
    plt.ylabel("Decoded output")
    plt.subplot(2, 1, 2)
    plt.xlim(right=t[-1])
    plt.ylabel("Neural activity")

    assert rmse(activity_data[0], activity_data[1]) < ((1. / dt) * 0.01)
    assert np.allclose(out_data[0], out_data[1], atol=0.05)


def test_amplitude(Simulator, seed, rng, plt):
    amp = 0.1
    neuron0 = nengo.LIF()
    neuron = nengo.LIF(amplitude=amp)

    # check static
    x = np.linspace(-5, 30)
    y = amp * neuron0.rates(x, 1., 0.)
    y2 = neuron.rates(x, 1., 0.)
    assert np.allclose(y, y2, atol=1e-5)

    # check dynamic
    n = 100
    x = 1.0
    encoders = np.ones((n, 1))
    max_rates = rng.uniform(low=20, high=200, size=n)
    intercepts = rng.uniform(low=-1, high=0.8, size=n)

    with nengo.Network(seed=seed) as model:
        ins = nengo.Node(x)
        ens = nengo.Ensemble(
            n, dimensions=1, neuron_type=neuron,
            encoders=encoders, max_rates=max_rates, intercepts=intercepts)
        nengo.Connection(ins, ens, synapse=None)
        spike_probe = nengo.Probe(ens.neurons)

    dt = 0.001
    t_final = 0.5
    with Simulator(model, dt=dt) as sim:
        sim.run(t_final)

    spikes = sim.data[spike_probe]
    r = spikes.sum(axis=0) * (dt / t_final)

    gain, bias = neuron0.gain_bias(max_rates, intercepts)
    r0 = amp * neuron0.rates(x, gain, bias)

    i = np.argsort(r0)
    plt.plot(r0[i])
    plt.plot(r[i])

    error = rms(r - r0) / rms(r0)
    assert (error < 0.02).all()


def test_reset(Simulator, nl_nodirect, seed, rng):
    """Make sure resetting actually resets."""
    m = nengo.Network(seed=seed)
    with m:
        m.config[nengo.Ensemble].neuron_type = nl_nodirect()
        u = nengo.Node(WhiteSignal(0.3, high=10), size_out=2)
        ens = nengo.Ensemble(60, dimensions=2)
        square = nengo.Ensemble(60, dimensions=2)
        nengo.Connection(u, ens)
        nengo.Connection(ens, square, function=lambda x: x**2,
                         solver=LstsqL2nz(weights=True))
        square_p = nengo.Probe(square, synapse=0.01)

    with Simulator(m, seed=seed+1) as sim:
        sim.run(0.1)
        sim.run(0.2)
        t1 = sim.trange()
        square1 = np.array(sim.data[square_p])

        sim.reset()
        sim.run(0.3)

    assert np.allclose(sim.trange(), t1)
    assert np.allclose(sim.data[square_p], square1)


def test_neurontypeparam():
    """NeuronTypeParam must be a neuron type."""
    class Test(object):
        ntp = NeuronTypeParam('ntp', default=None)

    inst = Test()
    inst.ntp = nengo.LIF()
    assert isinstance(inst.ntp, nengo.LIF)
    with pytest.raises(ValueError):
        inst.ntp = 'a'


def test_frozen():
    """Test attributes inherited from FrozenObject"""
    a = nengo.LIF()
    b = nengo.LIF()
    c = nengo.LIF(tau_rc=0.3)
    d = nengo.neurons.Izhikevich()

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


def test_direct_mode_nonfinite_value(Simulator):
    with nengo.Network() as model:
        e1 = nengo.Ensemble(10, 1, neuron_type=Direct())
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e1, e2, function=lambda x: 1. / x)

    with Simulator(model) as sim:
        with pytest.raises(SimulationError):
            sim.run(0.01)


@pytest.mark.parametrize("generic", (True, False))
def test_gain_bias(rng, nl_nodirect, generic):
    if nl_nodirect == nengo.Sigmoid and generic:
        # the generic method doesn't work with sigmoid neurons (because they're
        # always positive). that's not a failure, because the sigmoid neurons
        # never need to use the generic method normally, so we'll just skip
        # it for this test.
        return

    n = 100
    max_rates = rng.uniform(300, 400, size=n)
    intercepts = rng.uniform(-0.5, 0.5, size=n)
    nl = nl_nodirect()
    tolerance = 0.1 if generic else 1e-8

    if generic:
        gain, bias = nengo.neurons.NeuronType.gain_bias(nl, max_rates,
                                                        intercepts)
    else:
        gain, bias = nl.gain_bias(max_rates, intercepts)

    assert np.allclose(nl.rates(np.ones(n), gain, bias), max_rates,
                       atol=tolerance)

    if nl_nodirect == nengo.Sigmoid:
        threshold = 0.5 / nl.tau_ref
    else:
        threshold = 0

    assert np.all(nl.rates(intercepts - tolerance, gain, bias) <= threshold)
    assert np.all(nl.rates(intercepts + tolerance, gain, bias) > threshold)

    if generic:
        max_rates0, intercepts0 = (
            nengo.neurons.NeuronType.max_rates_intercepts(nl, gain, bias))
    else:
        max_rates0, intercepts0 = nl.max_rates_intercepts(gain, bias)

    assert np.allclose(max_rates, max_rates0, atol=tolerance)
    assert np.allclose(intercepts, intercepts0, atol=tolerance)
