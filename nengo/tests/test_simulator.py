import numpy as np
import pytest

import nengo
import nengo.simulator
from nengo.exceptions import SimulatorClosed
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
