import logging
import pickle
import pkg_resources

import numpy as np
import pytest

import nengo
import nengo.simulator
from nengo.builder import Model
from nengo.builder.ensemble import BuiltEnsemble
from nengo.builder.operator import DotInc
from nengo.builder.signal import Signal
from nengo.exceptions import SimulatorClosed, ValidationError
from nengo.rc import rc, RC_DEFAULTS
from nengo.utils.progress import ProgressBar


def test_steps(Simulator, allclose):
    """Tests stepping through a simple simulation, ensuring that
    steps are tracked and steps take the right amount of time"""
    dt = 0.001
    m = nengo.Network(label="test_steps")
    with Simulator(m, dt=dt) as sim:
        assert sim.n_steps == 0
        assert allclose(sim.time, 0 * dt)
        sim.step()
        assert sim.n_steps == 1
        assert allclose(sim.time, 1 * dt)
        sim.step()
        assert sim.n_steps == 2
        assert allclose(sim.time, 2 * dt)

        assert np.isscalar(sim.n_steps)
        assert np.isscalar(sim.time)


@pytest.mark.parametrize("bits", ["16", "32", "64"])
def test_dtype(Simulator, request, seed, bits):
    # Ensures dtype is set back to default after the test, even if it fails
    request.addfinalizer(
        lambda: rc.set("precision", "bits", str(RC_DEFAULTS["precision"]["bits"]))
    )

    float_dtype = np.dtype(getattr(np, "float%s" % bits))
    int_dtype = np.dtype(getattr(np, "int%s" % bits))

    with nengo.Network() as model:
        u = nengo.Node([0.5, -0.4])
        a = nengo.Ensemble(10, 2)
        nengo.Connection(u, a)
        p = nengo.Probe(a)

    rc["precision"]["bits"] = bits
    with Simulator(model) as sim:
        sim.step()

        for k, v in sim.signals.items():
            assert v.dtype in (float_dtype, int_dtype), "Signal '%s' wrong dtype" % k

        objs = (obj for obj in model.all_objects if sim.data[obj] is not None)
        for obj in objs:
            for x in (x for x in sim.data[obj] if isinstance(x, np.ndarray)):
                assert x.dtype == float_dtype, obj

        assert sim.data[p].dtype == float_dtype


def test_time_absolute(Simulator, allclose):
    m = nengo.Network()
    with Simulator(m) as sim:
        sim.run(0.003)
    assert allclose(sim.trange(), [0.001, 0.002, 0.003])


def test_trange_with_probes(Simulator):
    dt = 1e-3
    m = nengo.Network()
    periods = dt * np.arange(1, 21)
    with m:
        u = nengo.Node(output=np.sin)
        probes = [nengo.Probe(u, sample_every=p, synapse=5 * p) for p in periods]

    with Simulator(m, dt=dt) as sim:
        sim.run(0.333)
    for i, p in enumerate(periods):
        assert len(sim.trange(sample_every=p)) == len(sim.data[probes[i]])


def test_simulation_data():
    """Tests simulator.SimulationData's implementation."""
    raw = {"scalar": 5, "list": [2, 4, 6]}
    data = nengo.simulator.SimulationData(raw)
    assert np.all(data["scalar"] == np.asarray(raw["scalar"]))
    assert np.all(data.get("list") == np.asarray(raw.get("list")))


def test_simulation_data_with_repeated_simulator_runs(Simulator):
    with nengo.Network() as model:
        ens = nengo.Ensemble(10, 1)
        p = nengo.Probe(ens)

    dt = 0.001
    with Simulator(model, dt=dt) as sim:
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
        sim.run(1.0)
    with pytest.raises(SimulatorClosed):
        sim.reset()


def test_close_context(Simulator):
    m = nengo.Network()
    with m:
        nengo.Ensemble(10, 1)

    with Simulator(m) as sim:
        sim.run(0.01)

    with pytest.raises(SimulatorClosed):
        sim.run(1.0)
    with pytest.raises(SimulatorClosed):
        sim.reset()


def test_close_steps(Simulator):
    """For Simulator, closed simulators should fail for ``step``"""
    m = nengo.Network()
    with m:
        nengo.Ensemble(10, 1)

    # test close function
    sim = Simulator(m)
    sim.close()
    with pytest.raises(SimulatorClosed):
        sim.run_steps(1)
    with pytest.raises(SimulatorClosed):
        sim.step()

    # test close context
    with Simulator(m) as sim:
        sim.run(0.01)

    with pytest.raises(SimulatorClosed):
        sim.run_steps(1)
    with pytest.raises(SimulatorClosed):
        sim.step()


def test_warn_on_opensim_del(Simulator):
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)

    sim = Simulator(net)
    with pytest.warns(ResourceWarning):
        sim.__del__()
    sim.close()


def test_entry_point(Simulator):
    sims = [ep.load() for ep in pkg_resources.iter_entry_points(group="nengo.backends")]
    assert Simulator in sims


def test_signal_init_values(Simulator):
    """Tests that initial values are not overwritten."""
    zero = Signal([0])
    one = Signal([1])
    five = Signal([5.0])
    zeroarray = Signal([[0], [0], [0]])
    array = Signal([1, 2, 3])

    m = Model(dt=0)
    m.operators += [DotInc(zero, zero, five), DotInc(zeroarray, one, array)]

    with Simulator(None, model=m) as sim:
        assert sim.signals[zero][0] == 0
        assert sim.signals[one][0] == 1
        assert sim.signals[five][0] == 5.0
        assert np.all(np.array([1, 2, 3]) == sim.signals[array])
        sim.step()
        assert sim.signals[zero][0] == 0
        assert sim.signals[one][0] == 1
        assert sim.signals[five][0] == 5.0
        assert np.all(np.array([1, 2, 3]) == sim.signals[array])


def test_seeding(Simulator, allclose):
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
    with Simulator(m) as sim:
        m1 = sim.data
    with Simulator(m) as sim:
        m2 = sim.data
    m.seed = 873
    with Simulator(m) as sim:
        m3 = sim.data

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = allclose(getattr(obj1, attr), getattr(obj2, attr)) == equal
            if not check:
                logging.info("%s: %s", attr, getattr(obj1, attr))
                logging.info("%s: %s", attr, getattr(obj2, attr))
            assert check

    ens_attrs = BuiltEnsemble._fields
    As = [mi[A] for mi in [m1, m2, m3]]
    Bs = [mi[B] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], ens_attrs)
    compare_objs(Bs[0], Bs[1], ens_attrs)
    compare_objs(As[0], As[2], ens_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], ens_attrs, equal=False)

    conn_attrs = ("eval_points", "weights")
    Cs = [mi[C] for mi in [m1, m2, m3]]
    compare_objs(Cs[0], Cs[1], conn_attrs)
    compare_objs(Cs[0], Cs[2], conn_attrs, equal=False)


def test_hierarchical_seeding():
    """Changes to subnetworks shouldn't affect seeds in top-level network"""

    def create(make_extra, seed):
        objs = []
        with nengo.Network(seed=seed, label="n1") as model:
            objs.append(nengo.Ensemble(10, 1, label="e1"))
            with nengo.Network(label="n2"):
                objs.append(nengo.Ensemble(10, 1, label="e2"))
                if make_extra:
                    # This shouldn't affect any seeds
                    objs.append(nengo.Ensemble(10, 1, label="e3"))
            objs.append(nengo.Ensemble(10, 1, label="e4"))
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


def test_probe_cache(Simulator, allclose):
    with nengo.Network() as model:
        u = nengo.Node(nengo.processes.WhiteNoise())
        up = nengo.Probe(u)

    with Simulator(model, seed=0) as sim:
        sim.run_steps(10)
        ua = np.array(sim.data[up])

        sim.reset(seed=1)
        sim.run_steps(10)
        ub = np.array(sim.data[up])

    assert not allclose(ua, ub, atol=1e-1, record_rmse=False)


def test_invalid_run_time(Simulator):
    net = nengo.Network()
    with Simulator(net) as sim:
        with pytest.raises(ValidationError):
            sim.run(-0.0001)
        with pytest.warns(UserWarning):
            sim.run(0)
        sim.run(0.0006)  # Rounds up to 0.001
        assert sim.n_steps == 1


def test_sim_seed_set_by_network_seed(Simulator, seed):
    with nengo.Network(seed=seed) as model:
        pass
    with Simulator(model) as sim:
        sim_seed = sim.seed
    with Simulator(model) as sim:
        assert sim.seed == sim_seed


def test_simulator_progress_bars(Simulator):
    class ProgressBarInvariants(ProgressBar):
        def __init__(self):
            super().__init__()
            self.progress = None
            self.max_steps = None
            self.n_steps = 0
            self.closed = False

        def update(self, progress):
            assert not self.closed
            if self.progress is not progress:
                assert self.max_steps is None or self.n_steps <= self.max_steps
                self.n_steps = progress.n_steps
                self.max_steps = progress.max_steps
                self.progress = progress
            assert progress.max_steps == self.max_steps
            assert self.max_steps is None or self.n_steps <= progress.n_steps
            self.n_steps = progress.n_steps

        def close(self):
            self.closed = True

    with nengo.Network() as model:
        for _ in range(3):
            for _ in range(3):
                nengo.Ensemble(10, 1)
            with nengo.Network():
                for _ in range(3):
                    nengo.Ensemble(10, 1)
    build_invariants = ProgressBarInvariants()
    with Simulator(model, progress_bar=build_invariants) as sim:
        run_invariants = ProgressBarInvariants()
        sim.run(0.01, progress_bar=run_invariants)
        assert run_invariants.n_steps == run_invariants.max_steps


@pytest.mark.parametrize("sample_every", (0.001, 0.0005, 0.002, 0.0015))
def test_sample_every_trange(Simulator, sample_every, allclose):
    with nengo.Network() as model:
        t = nengo.Node(lambda t: t)
        p = nengo.Probe(t, sample_every=sample_every)

    with Simulator(model) as sim:
        sim.run(0.01)

    with pytest.raises(ValidationError):
        sim.trange(dt=sample_every, sample_every=sample_every)
    with pytest.warns(DeprecationWarning):
        assert allclose(sim.trange(dt=sample_every), np.squeeze(sim.data[p]))
    assert allclose(sim.trange(sample_every=sample_every), np.squeeze(sim.data[p]))


def test_pickle_optimize(caplog, seed):
    caplog.set_level(logging.DEBUG)
    with nengo.Network(seed=seed) as net:
        stim = nengo.Node(np.zeros(10))
        ea = nengo.networks.EnsembleArray(20, stim.size_out)
        nengo.Connection(stim, ea.input)
        net.probe = nengo.Probe(ea.output)

    with nengo.Simulator(net, optimize=True) as sim:
        pickled = pickle.dumps(sim)
        sim.run(0.01)
        assert not sim.closed
        assert sim.optimize

    before = sim.data[net.probe]
    del net, sim

    # Check that original sim was optimized
    assert any(record.msg == "Optimizing model..." for record in caplog.records)
    caplog.clear()

    unpickled = pickle.loads(pickled)
    net = unpickled.model.toplevel
    assert not unpickled.closed
    assert unpickled.optimize

    # Check that unpickling doesn't re-do optimization
    assert not any(record.msg == "Optimizing model..." for record in caplog.records)

    unpickled.run(0.01)
    after = unpickled.data[net.probe]
    unpickled.close()

    assert np.all(before == after)
