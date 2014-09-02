import logging

import numpy as np
import pytest

import nengo
from nengo.utils.compat import range
from nengo.utils.testing import Timer

logger = logging.getLogger(__name__)


def test_multirun(Simulator):
    """Test probing the time on multiple runs"""
    rng = np.random.RandomState(2239)

    # set rtol a bit higher, since OCL model.t accumulates error over time
    rtol = 0.0001

    model = nengo.Network(label="Multi-run")

    sim = Simulator(model)

    # t_stops = [0.123, 0.283, 0.821, 0.921]
    t_stops = sim.dt * rng.randint(low=100, high=2000, size=10)

    t_sum = 0
    for ti in t_stops:
        sim.run(ti)
        sim_t = sim.trange()
        t = sim.dt * np.arange(len(sim_t))
        assert np.allclose(sim_t, t, rtol=rtol)

        t_sum += ti
        assert np.allclose(sim_t[-1], t_sum - sim.dt, rtol=rtol)


def test_dts(Simulator):
    """Test probes with different sampling times."""

    n = 10

    rng = np.random.RandomState(48392)
    dts = 0.001 * rng.randint(low=1, high=100, size=n)
    # dts = 0.001 * np.hstack([2, rng.randint(low=1, high=100, size=n-1)])

    def input_fn(t):
        return list(range(1, 10))

    model = nengo.Network(label='test_probe_dts', seed=2891)
    with model:
        probes = []
        for i, dt in enumerate(dts):
            xi = nengo.Node(label='x%d' % i, output=input_fn)
            p = nengo.Probe(xi, 'output', sample_every=dt)
            probes.append(p)

    sim = Simulator(model)
    simtime = 2.483
    # simtime = 2.484

    with Timer() as timer:
        sim.run(simtime)
    logger.debug("Ran %d probes for %f sec simtime in %0.3f sec",
                 n, simtime, timer.duration)

    for i, p in enumerate(probes):
        t = sim.dt * np.arange(int(np.ceil(simtime / dts[i])))
        x = np.asarray([input_fn(tt) for tt in t])
        y = sim.data[p]
        assert len(x) == len(y)
        assert np.allclose(y[1:], x[:-1])  # 1-step delay


def test_large(Simulator):
    """Test with a lot of big probes. Can also be used for speed."""

    n = 10

    def input_fn(t):
        return list(range(1, 10))

    model = nengo.Network(label='test_large_probes', seed=3249)
    with model:
        probes = []
        for i in range(n):
            xi = nengo.Node(label='x%d' % i, output=input_fn)
            probes.append(nengo.Probe(xi, 'output'))

    sim = Simulator(model)
    simtime = 2.483

    with Timer() as timer:
        sim.run(simtime)
    logger.debug("Ran %d probes for %f sec simtime in %0.3f sec",
                 n, simtime, timer.duration)

    t = sim.dt * np.arange(int(np.round(simtime / sim.dt)))
    x = np.asarray([input_fn(ti) for ti in t])
    for p in probes:
        y = sim.data[p]
        assert np.allclose(y[1:], x[:-1])  # 1-step delay


def test_defaults(Simulator):
    """Tests that probing with no attr sets the right attr."""
    model = nengo.Network(label='test_defaults')
    with model:
        node = nengo.Node(output=0.5)
        ens = nengo.Ensemble(20, 1)
        conn = nengo.Connection(node, ens)
        node_p = nengo.Probe(node)
        assert node_p.attr == 'output'
        ens_p = nengo.Probe(ens)
        assert ens_p.attr == 'decoded_output'
        conn_p = nengo.Probe(conn)
        assert conn_p.attr == 'signal'
    # Let's just make sure it runs too...
    sim = Simulator(model)
    sim.run(0.01)


def test_simulator_dt(Simulator):
    """Changing the simulator dt should change the default probe dt."""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        nengo.Connection(a, b)
        bp = nengo.Probe(b)

    sim = nengo.Simulator(model, dt=0.01)
    sim.run(1.)
    assert sim.data[bp].shape == (100, 1)


def test_multiple_probes(Simulator):
    """Make sure we can probe the same object multiple times."""
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(10, 1)
        p_001 = nengo.Probe(ens, sample_every=0.001)
        p_01 = nengo.Probe(ens, sample_every=0.01)
        p_1 = nengo.Probe(ens, sample_every=0.1)

    sim = nengo.Simulator(model, dt=0.001)
    sim.run(1.)
    assert np.allclose(sim.data[p_001][::10], sim.data[p_01])
    assert np.allclose(sim.data[p_01][::10], sim.data[p_1])


def test_input_probe(Simulator):
    """Make sure we can probe the input to an ensemble."""
    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(100, 1)
        n1 = nengo.Node(output=np.sin)
        n2 = nengo.Node(output=0.5)
        nengo.Connection(n1, ens, synapse=None)
        nengo.Connection(n2, ens, synapse=None)
        input_probe = nengo.Probe(ens, 'input', synapse=None)

        sim = nengo.Simulator(model)
        sim.run(1.)
        t = sim.trange()
        assert np.allclose(sim.data[input_probe][1:, 0], (np.sin(t) + 0.5)[1:])


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
