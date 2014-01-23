import numpy as np
import pytest

import nengo
from nengo import builder
from nengo.tests.helpers import rms

import logging
logger = logging.getLogger(__name__)


def mybuilder(model, dt):
    model.dt = dt
    model.seed = 0
    if not hasattr(model, 'probes'):
        model.probes = []
    return model


def test_lif_builtin():
    """Test that the dynamic model approximately matches the rates."""
    rng = np.random.RandomState(85243)

    lif = nengo.LIF(10)
    lif.set_gain_bias(rng.uniform(80, 100, (10,)), rng.uniform(-1, 1, (10,)))
    J = np.arange(0, 5, .5)
    voltage = np.zeros(10)
    reftime = np.zeros(10)

    spikes = []
    spikes_ii = np.zeros(10)

    for ii in range(1000):
        lif.step_math0(.001, J + lif.bias, voltage, reftime, spikes_ii)
        spikes.append(spikes_ii.copy())

    sim_rates = np.sum(spikes, axis=0)
    math_rates = lif.rates(J)
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)


def test_pyfunc():
    """Test Python Function nonlinearity"""
    dt = 0.001
    d = 3
    n_steps = 3
    n_trials = 3

    rng = np.random.RandomState(seed=987)

    for i in range(n_trials):
        A = rng.normal(size=(d, d))
        fn = lambda t, x: np.cos(np.dot(A, x))

        x = np.random.normal(size=d)

        m = nengo.Model("")
        ins = builder.Signal(x, name='ins')
        pop = nengo.PythonFunction(fn=fn, n_in=d, n_out=d)
        m.operators = []
        b = builder.Builder()
        b.model = m
        b.build_pyfunc(pop)
        m.operators += [
            builder.DotInc(builder.Signal(np.eye(d)), ins, pop.input_signal),
            builder.ProdUpdate(builder.Signal(np.eye(d)),
                               pop.output_signal,
                               builder.Signal(0),
                               ins)
        ]

        sim = nengo.Simulator(m, dt=dt, builder=mybuilder)

        p0 = np.zeros(d)
        s0 = np.array(x)
        for j in range(n_steps):
            tmp = p0
            p0 = fn(0, s0)
            s0 = tmp
            sim.step()
            assert np.allclose(s0, sim.signals[ins])
            assert np.allclose(p0, sim.signals[pop.output_signal])


def test_lif_base(nl_nodirect):
    """Test that the dynamic model approximately matches the rates"""
    rng = np.random.RandomState(85243)

    dt = 0.001
    d = 1
    n = 5000

    m = nengo.Model("")
    ins = builder.Signal(0.5 * np.ones(d), name='ins')
    lif = nl_nodirect(n)
    lif.set_gain_bias(max_rates=rng.uniform(low=10, high=200, size=n),
                      intercepts=rng.uniform(low=-1, high=1, size=n))
    m.operators = []
    b = builder.Builder()
    b.model = m
    b._builders[nl_nodirect](lif)
    m.operators.append(builder.DotInc(
        builder.Signal(np.ones((n, d))), ins, lif.input_signal))

    sim = nengo.Simulator(m, dt=dt, builder=mybuilder)

    t_final = 1.0
    spikes = np.zeros(n)
    for i in range(int(np.round(t_final / dt))):
        sim.step()
        spikes += sim.signals[lif.output_signal]

    math_rates = lif.rates(sim.signals[lif.input_signal] - lif.bias)
    sim_rates = spikes / t_final
    logger.debug("ME = %f", (sim_rates - math_rates).mean())
    logger.debug("RMSE = %f",
                 rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
    assert np.sum(math_rates > 0) > 0.5 * n, (
        "At least 50% of neurons must fire")
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
