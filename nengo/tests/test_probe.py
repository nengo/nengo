import logging
import time

import numpy as np
import pytest

import nengo

logger = logging.getLogger(__name__)


def test_multirun(Simulator):
    """Test probing the time on multiple runs"""
    rng = np.random.RandomState(2239)

    # set rtol a bit higher, since OCL model.t accumulates error over time
    rtol = 0.0001

    model = nengo.Model("Multi-run")

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

    model = nengo.Model('test_probe_dts', seed=2891)

    probes = []
    for i, dt in enumerate(dts):
        xi = nengo.Node(label='x%d' % i, output=input_fn)
        p = nengo.Probe(xi, 'output', dt=dt)
        probes.append(p)

    sim = Simulator(model)
    simtime = 2.483
    # simtime = 2.484

    timer = time.time()
    sim.run(simtime)
    timer = time.time() - timer
    logger.debug(
        "Ran %(n)s probes for %(simtime)s sec simtime in %(timer)0.3f sec"
        % locals())

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

    model = nengo.Model('test_large_probes', seed=3249)

    probes = []
    for i in range(n):
        xi = nengo.Node(label='x%d' % i, output=input_fn)
        probes.append(nengo.Probe(xi, 'output'))

    sim = Simulator(model)
    simtime = 2.483

    timer = time.time()
    sim.run(simtime)
    timer = time.time() - timer
    logger.debug(
        "Ran %(n)s probes for %(simtime)s sec simtime in %(timer)0.3f sec"
        % locals())

    t = sim.dt * np.arange(int(np.round(simtime / sim.dt)))
    x = np.asarray([input_fn(ti) for ti in t])
    for p in probes:
        y = sim.data[p]
        assert np.allclose(y[1:], x[:-1])  # 1-step delay


def test_defaults(Simulator):
    """Tests that probing with no attr sets the right attr."""
    model = nengo.Model('test_defaults')
    node = nengo.Node(output=0.5)
    ens = nengo.Ensemble(nengo.LIF(20), 1)
    conn = nengo.Connection(node, ens)
    node_p = nengo.Probe(node)
    assert node_p.attr == 'output'
    ens_p = nengo.Probe(ens)
    assert ens_p.attr == 'decoded_output'
    with pytest.raises(TypeError):
        nengo.Probe(conn)
    # Let's just make sure it runs too...
    sim = Simulator(model)
    sim.run(0.01)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
