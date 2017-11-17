import numpy as np
import pkg_resources
import pytest

import nengo
import nengo.simulator
from nengo.builder import Model
from nengo.builder.ensemble import BuiltEnsemble
from nengo.builder.operator import DotInc
from nengo.builder.signal import Signal
from nengo.exceptions import ObsoleteError, SimulatorClosed, ValidationError
from nengo.utils.compat import ResourceWarning
from nengo.utils.testing import warns


def test_steps(RefSimulator):
    dt = 0.001
    m = nengo.Network(label="test_steps")
    with RefSimulator(m, dt=dt) as sim:
        assert sim.n_steps == 0
        assert np.allclose(sim.time, 0 * dt)
        sim.step()
        assert sim.n_steps == 1
        assert np.allclose(sim.time, 1 * dt)
        sim.step()
        assert sim.n_steps == 2
        assert np.allclose(sim.time, 2 * dt)


def test_time_absolute(Simulator):
    m = nengo.Network()
    with Simulator(m) as sim:
        sim.run(0.003)
    assert np.allclose(sim.trange(), [0.001, 0.002, 0.003])


def test_trange_with_probes(Simulator):
    dt = 1e-3
    m = nengo.Network()
    periods = dt * np.arange(1, 21)
    with m:
        u = nengo.Node(output=np.sin)
        probes = [nengo.Probe(u, sample_every=p, synapse=5*p) for p in periods]

    with Simulator(m, dt=dt) as sim:
        sim.run(0.333)
    for i, p in enumerate(periods):
        assert len(sim.trange(p)) == len(sim.data[probes[i]])


def test_probedict():
    """Tests simulator.ProbeDict's implementation."""
    raw = {"scalar": 5,
           "list": [2, 4, 6]}
    probedict = nengo.simulator.ProbeDict(raw)
    assert np.all(probedict["scalar"] == np.asarray(raw["scalar"]))
    assert np.all(probedict.get("list") == np.asarray(raw.get("list")))


def test_probedict_with_repeated_simulator_runs(RefSimulator):
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)
        p = nengo.Probe(ens)

    dt = 0.001
    with RefSimulator(model, dt=dt) as sim:
        sim.run(0.01)
        assert len(sim.data[p]) == 10
        sim.run(0.01)
        assert len(sim.data[p]) == 20


def test_close_function(Simulator):
    m = nengo.Network()
    with m:
        nengo.Ensemble(10, 1)

    sim = Simulator(m)
    sim.close()
    with pytest.raises(SimulatorClosed):
        sim.run(1.)
    with pytest.raises(SimulatorClosed):
        sim.reset()


def test_close_context(Simulator):
    m = nengo.Network()
    with m:
        nengo.Ensemble(10, 1)

    with Simulator(m) as sim:
        sim.run(0.01)

    with pytest.raises(SimulatorClosed):
        sim.run(1.)
    with pytest.raises(SimulatorClosed):
        sim.reset()


def test_close_steps(RefSimulator):
    """For RefSimulator, closed simulators should fail for ``step``"""
    m = nengo.Network()
    with m:
        nengo.Ensemble(10, 1)

    # test close function
    sim = RefSimulator(m)
    sim.close()
    with pytest.raises(SimulatorClosed):
        sim.run_steps(1)
    with pytest.raises(SimulatorClosed):
        sim.step()

    # test close context
    with RefSimulator(m) as sim:
        sim.run(0.01)

    with pytest.raises(SimulatorClosed):
        sim.run_steps(1)
    with pytest.raises(SimulatorClosed):
        sim.step()


def test_warn_on_opensim_del(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)

    sim = Simulator(net)
    with warns(ResourceWarning):
        sim.__del__()
    sim.close()


def test_entry_point(Simulator):
    sims = [ep.load() for ep in
            pkg_resources.iter_entry_points(group='nengo.backends')]
    assert Simulator in sims


def test_signal_init_values(RefSimulator):
    """Tests that initial values are not overwritten."""
    zero = Signal([0])
    one = Signal([1])
    five = Signal([5.0])
    zeroarray = Signal([[0], [0], [0]])
    array = Signal([1, 2, 3])

    m = Model(dt=0)
    m.operators += [DotInc(zero, zero, five), DotInc(zeroarray, one, array)]

    with RefSimulator(None, model=m) as sim:
        assert sim.signals[zero][0] == 0
        assert sim.signals[one][0] == 1
        assert sim.signals[five][0] == 5.0
        assert np.all(np.array([1, 2, 3]) == sim.signals[array])
        sim.step()
        assert sim.signals[zero][0] == 0
        assert sim.signals[one][0] == 1
        assert sim.signals[five][0] == 5.0
        assert np.all(np.array([1, 2, 3]) == sim.signals[array])


def test_seeding(RefSimulator, logger):
    """Test that setting the model seed fixes everything"""

    #  TODO: this really just checks random parameters in ensembles.
    #   Are there other objects with random parameters that should be
    #   tested? (Perhaps initial weights of learned connections)

    m = nengo.Network(label="test_seeding")
    with m:
        input = nengo.Node(output=1, label="input")
        A = nengo.Ensemble(40, 1, label="A")
        B = nengo.Ensemble(20, 1, label="B")
        nengo.Connection(input, A)
        C = nengo.Connection(A, B, function=lambda x: x ** 2)

    m.seed = 872
    with RefSimulator(m) as sim:
        m1 = sim.data
    with RefSimulator(m) as sim:
        m2 = sim.data
    m.seed = 873
    with RefSimulator(m) as sim:
        m3 = sim.data

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = (np.allclose(getattr(obj1, attr), getattr(obj2, attr)) ==
                     equal)
            if not check:
                logger.info("%s: %s", attr, getattr(obj1, attr))
                logger.info("%s: %s", attr, getattr(obj2, attr))
            assert check

    ens_attrs = BuiltEnsemble._fields
    As = [mi[A] for mi in [m1, m2, m3]]
    Bs = [mi[B] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], ens_attrs)
    compare_objs(Bs[0], Bs[1], ens_attrs)
    compare_objs(As[0], As[2], ens_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], ens_attrs, equal=False)

    conn_attrs = ('eval_points', 'weights')
    Cs = [mi[C] for mi in [m1, m2, m3]]
    compare_objs(Cs[0], Cs[1], conn_attrs)
    compare_objs(Cs[0], Cs[2], conn_attrs, equal=False)


def test_hierarchical_seeding():
    """Changes to subnetworks shouldn't affect seeds in top-level network"""

    def create(make_extra, seed):
        objs = []
        with nengo.Network(seed=seed, label='n1') as model:
            objs.append(nengo.Ensemble(10, 1, label='e1'))
            with nengo.Network(label='n2'):
                objs.append(nengo.Ensemble(10, 1, label='e2'))
                if make_extra:
                    # This shouldn't affect any seeds
                    objs.append(nengo.Ensemble(10, 1, label='e3'))
            objs.append(nengo.Ensemble(10, 1, label='e4'))
        return model, objs

    same1, same1objs = create(False, 9)
    same2, same2objs = create(True, 9)
    diff, diffobjs = create(True, 10)

    m1 = Model()
    m1.build(same1)
    same1seeds = m1.seeds

    m2 = Model()
    m2.build(same2)
    same2seeds = m2.seeds

    m3 = Model()
    m3.build(diff)
    diffseeds = m3.seeds

    for diffobj, same2obj in zip(diffobjs, same2objs):
        # These seeds should all be different
        assert diffseeds[diffobj] != same2seeds[same2obj]

    # Skip the extra ensemble
    same2objs = same2objs[:2] + same2objs[3:]

    for same1obj, same2obj in zip(same1objs, same2objs):
        # These seeds should all be the same
        assert same1seeds[same1obj] == same2seeds[same2obj]


def test_obsolete_params(RefSimulator):
    with nengo.Network() as net:
        e = nengo.Ensemble(10, 1)
        c = nengo.Connection(e, e)
    with RefSimulator(net) as sim:
        pass
    with pytest.raises(ObsoleteError):
        sim.data[c].decoders


def test_probe_cache(Simulator):
    with nengo.Network() as model:
        u = nengo.Node(nengo.processes.WhiteNoise())
        up = nengo.Probe(u)

    with Simulator(model, seed=0) as sim:
        sim.run_steps(10)
        ua = np.array(sim.data[up])

        sim.reset(seed=1)
        sim.run_steps(10)
        ub = np.array(sim.data[up])

    assert not np.allclose(ua, ub, atol=1e-1)


def test_invalid_run_time(Simulator):
    net = nengo.Network()
    with Simulator(net) as sim:
        with pytest.raises(ValidationError):
            sim.run(-0.0001)
        with warns(UserWarning):
            sim.run(0)
        sim.run(0.0006)  # Rounds up to 0.001
        assert sim.n_steps == 1


def test_sim_seed_set_by_network_seed(Simulator, seed):
    with nengo.Network(seed=seed) as model:
        pass
    with nengo.Simulator(model) as sim:
        sim_seed = sim.seed
    with nengo.Simulator(model) as sim:
        assert sim.seed == sim_seed
