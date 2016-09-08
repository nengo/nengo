import numpy as np
import pytest

import nengo
from nengo.exceptions import ObsoleteError
from nengo.utils.compat import range
from nengo.utils.stdlib import Timer


def test_multirun(Simulator, rng):
    """Test probing the time on multiple runs"""

    # set rtol a bit higher, since OCL model.t accumulates error over time
    rtol = 0.0001

    model = nengo.Network(label="Multi-run")

    with Simulator(model) as sim:
        # t_stops = [0.123, 0.283, 0.821, 0.921]
        t_stops = sim.dt * rng.randint(low=100, high=2000, size=10)

        t_sum = 0
        for ti in t_stops:
            sim.run(ti)
            sim_t = sim.trange()
            t = sim.dt * np.arange(1, len(sim_t) + 1)
            assert np.allclose(sim_t, t, rtol=rtol)

            t_sum += ti
            assert np.allclose(sim_t[-1], t_sum, rtol=rtol)


def test_dts(Simulator, seed, rng):
    """Test probes with different dts and runtimes"""

    for i in range(100):
        dt = rng.uniform(0.001, 0.1)  # simulator dt
        dt2 = rng.uniform(dt, 0.15)   # probe dt
        tend = rng.uniform(0.2, 0.3)  # simulator runtime

        with nengo.Network(seed=seed) as model:
            a = nengo.Node(output=0)
            ap = nengo.Probe(a, sample_every=dt2)

        with Simulator(model, dt=dt) as sim:
            sim.run(tend)
        t = sim.trange(dt2)
        x = sim.data[ap]

        assert len(t) == len(x), "dt=%f, dt2=%f, tend=%f, nt=%d, nx=%d" % (
            dt, dt2, tend, len(t), len(x))


def test_large(Simulator, seed, logger):
    """Test with a lot of big probes. Can also be used for speed."""

    n = 10

    def input_fn(t):
        return list(range(1, 10))

    model = nengo.Network(label='test_large_probes', seed=seed)
    with model:
        probes = []
        for i in range(n):
            xi = nengo.Node(label='x%d' % i, output=input_fn)
            probes.append(nengo.Probe(xi, 'output'))

    with Simulator(model) as sim:
        simtime = 2.483

        with Timer() as timer:
            sim.run(simtime)
    logger.info("Ran %d probes for %f sec simtime in %0.3f sec",
                n, simtime, timer.duration)

    t = sim.trange()
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
        assert conn_p.attr == 'output'
    # Let's just make sure it runs too...
    with Simulator(model) as sim:
        sim.run(0.01)


def test_simulator_dt(Simulator):
    """Changing the simulator dt should change the default probe dt."""
    with nengo.Network() as model:
        a = nengo.Ensemble(10, 1)
        b = nengo.Ensemble(10, 1)
        nengo.Connection(a, b)
        bp = nengo.Probe(b)

    with Simulator(model, dt=0.01) as sim:
        sim.run(1.)
    assert sim.data[bp].shape == (100, 1)


def test_multiple_probes(Simulator):
    """Make sure we can probe the same object multiple times."""
    dt = 1e-3
    f = 10
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)
        p_001 = nengo.Probe(ens, sample_every=dt)
        p_01 = nengo.Probe(ens, sample_every=f * dt)
        p_1 = nengo.Probe(ens, sample_every=f**2 * dt)

    with Simulator(model, dt=dt) as sim:
        sim.run(1.)
    assert np.allclose(sim.data[p_001][f - 1::f], sim.data[p_01])
    assert np.allclose(sim.data[p_01][f - 1::f], sim.data[p_1])


def test_input_probe(Simulator):
    """Make sure we can probe the input to an ensemble."""
    with nengo.Network() as model:
        ens = nengo.Ensemble(100, 1)
        n1 = nengo.Node(output=np.sin)
        n2 = nengo.Node(output=0.5)
        nengo.Connection(n1, ens, synapse=None)
        nengo.Connection(n2, ens, synapse=None)
        input_probe = nengo.Probe(ens, 'input')

    with Simulator(model) as sim:
        sim.run(1.)
    t = sim.trange()
    assert np.allclose(sim.data[input_probe][:, 0], np.sin(t) + 0.5)


def test_conn_output(Simulator):
    """Make sure we can get individual connection outputs."""
    model = nengo.Network()
    with model:
        n1 = nengo.Node(output=np.sin)
        n2 = nengo.Node(output=0.5)
        n_out = nengo.Node(size_in=1)
        c1 = nengo.Connection(n1, n_out, transform=-1, synapse=None)
        c2 = nengo.Connection(n2, n_out, function=lambda x: x**2, synapse=None)
        p1 = nengo.Probe(c1, 'output', synapse=None)
        p2 = nengo.Probe(c2, 'output', synapse=None)

    with Simulator(model) as sim:
        sim.run(0.2)
    t = sim.trange()
    assert np.allclose(sim.data[p1][:, 0], -1. * np.sin(t))
    assert np.allclose(sim.data[p2][:, 0], 0.5 ** 2)


def test_slice(Simulator):
    with nengo.Network() as model:
        a = nengo.Node(output=lambda t: [np.cos(t), np.sin(t)])
        b = nengo.Ensemble(100, 2)
        nengo.Connection(a, b)

        bp = nengo.Probe(b, synapse=0.03)
        bp0a = nengo.Probe(b[0], synapse=0.03)
        bp0b = nengo.Probe(b[:1], synapse=0.03)
        bp1a = nengo.Probe(b[1], synapse=0.03)
        bp1b = nengo.Probe(b[1:], synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)
    assert np.allclose(sim.data[bp][:, 0], sim.data[bp0a][:, 0])
    assert np.allclose(sim.data[bp][:, 0], sim.data[bp0b][:, 0])
    assert np.allclose(sim.data[bp][:, 1], sim.data[bp1a][:, 0])
    assert np.allclose(sim.data[bp][:, 1], sim.data[bp1b][:, 0])


def test_solver_defaults():
    solver1 = nengo.solvers.LstsqL2(reg=0.764)
    solver2 = nengo.solvers.LstsqL2(reg=0.911)
    solver3 = nengo.solvers.LstsqL2(reg=0.898)

    make_probe = lambda: nengo.Probe(nengo.Ensemble(100, 1))

    with nengo.Network() as model:
        a = make_probe()
        model.config[nengo.Connection].solver = solver1
        b = make_probe()

        with nengo.Network() as net:
            c = make_probe()
            net.config[nengo.Probe].solver = solver2
            d = make_probe()

        net = nengo.Network()
        with net:
            e = make_probe()

        net.config[nengo.Probe].solver = solver3
        with net:
            f = make_probe()

    assert a.solver is nengo.Connection.solver.default
    assert b.solver is solver1
    assert c.solver is solver1
    assert d.solver is solver2
    assert e.solver is solver1
    assert f.solver is solver3


def test_ensemble_encoders(Simulator):
    """Check that encoders probed from ensemble are correct."""
    with nengo.Network() as model:
        ens = nengo.Ensemble(n_neurons=10, dimensions=2, radius=1.5)
        p_enc = nengo.Probe(ens, 'scaled_encoders')

    with Simulator(model) as sim:
        sim.run(0.001)

    ens_data = sim.data[ens]
    from_probe = sim.data[p_enc] / (ens_data.gain / ens.radius)[:, np.newaxis]
    from_data = ens_data.encoders
    assert np.allclose(from_probe, from_data)
    assert np.allclose(sim.data[p_enc], ens_data.scaled_encoders)


def test_obsolete_probes():
    with nengo.Network():
        pre = nengo.Ensemble(10, 1)
        post = nengo.Ensemble(10, 1)
        conn = nengo.Connection(pre, post)
        with pytest.raises(ObsoleteError):
            nengo.Probe(conn, "decoders")
        with pytest.raises(ObsoleteError):
            nengo.Probe(conn, "transform")
