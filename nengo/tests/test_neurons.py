import logging
import numpy as np
import pytest

import nengo
from nengo.neurons import NeuronTypeParam
from nengo.utils.ensemble import tuning_curves
from nengo.utils.matplotlib import implot
from nengo.utils.neurons import rates_kernel
from nengo.utils.numpy import rms

logger = logging.getLogger(__name__)


def test_lif_builtin():
    """Test that the dynamic model approximately matches the rates."""
    rng = np.random.RandomState(85243)

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
    sim_rates = spikes.sum(0)
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)


def test_lif(Simulator, plt):
    """Test that the dynamic model approximately matches the rates"""
    rng = np.random.RandomState(85243)

    dt = 0.001
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
        nengo.Connection(ins, ens.neurons, transform=np.ones((n, 1)))
        spike_probe = nengo.Probe(ens.neurons)
        voltage_probe = nengo.Probe(ens.neurons, 'voltage')
        ref_probe = nengo.Probe(ens.neurons, 'refractory_time')

    sim = Simulator(m, dt=dt)

    t_final = 1.0
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
    sim_rates = sim.data[spike_probe].sum(0) / t_final
    logger.debug("ME = %f", (sim_rates - math_rates).mean())
    logger.debug("RMSE = %f",
                 rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
    assert np.sum(math_rates > 0) > 0.5 * n, (
        "At least 50% of neurons must fire")
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)

    # if voltage and ref time are non-constant, the probe is doing something
    assert np.abs(np.diff(sim.data[voltage_probe])).sum() > 1
    assert np.abs(np.diff(sim.data[ref_probe])).sum() > 1


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
    sim = Simulator(model, dt=dt)
    sim.run(2.)

    t = sim.trange()
    rates = sim.data[ap]
    _, ref = tuning_curves(a, sim, eval_points=0.5)

    ax = plt.subplot(211)
    implot(plt, t, intercepts[::-1], rates.T / dt, ax=ax)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('input')
    ax = plt.subplot(212)
    ax.plot(intercepts, ref[:, ::-1].T, 'k--')
    ax.plot(intercepts, rates[[1, 500, 1000, -1], ::-1].T / dt)
    ax.set_xlabel('input')
    ax.set_xlabel('rate')

    # check that initial tuning curve is the same as LIF rates
    assert np.allclose(rates[1] / dt, ref, atol=0.1, rtol=1e-3)

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
    sim = Simulator(model, dt=dt)
    sim.run(2.)

    t = sim.trange()
    a_rates = sim.data[ap] / dt
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


def test_neurontypeparam():
    """NeuronTypeParam must be a neuron type."""
    class Test(object):
        ntp = NeuronTypeParam(default=None)

    inst = Test()
    inst.ntp = nengo.LIF()
    assert isinstance(inst.ntp, nengo.LIF)
    with pytest.raises(ValueError):
        inst.ntp = 'a'


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
