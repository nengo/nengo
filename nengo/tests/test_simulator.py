import logging

import numpy as np
import pytest

import nengo
import nengo.simulator
from nengo.builder import Model
from nengo.builder.node import build_pyfunc
from nengo.builder.operator import Copy, Reset, DotInc, SimNoise
from nengo.builder.signal import Signal
from nengo.utils.compat import range


logger = logging.getLogger(__name__)


def test_steps(RefSimulator):
    m = nengo.Network(label="test_steps")
    sim = RefSimulator(m)
    assert sim.n_steps == 0
    sim.step()
    assert sim.n_steps == 1
    sim.step()
    assert sim.n_steps == 2


def test_time_steps(RefSimulator):
    m = nengo.Network(label="test_time_steps")
    sim = RefSimulator(m)
    assert np.allclose(sim.signals["__time__"], 0.00)
    sim.step()
    assert np.allclose(sim.signals["__time__"], 0.001)
    sim.step()
    assert np.allclose(sim.signals["__time__"], 0.002)


def test_time_absolute(Simulator):
    m = nengo.Network()
    sim = Simulator(m)
    sim.run(0.003)
    assert np.allclose(sim.trange(), [0.001, 0.002, 0.003])


def test_trange_with_probes(Simulator):
    dt = 1e-3
    m = nengo.Network()
    periods = dt * np.arange(1, 21)
    with m:
        u = nengo.Node(output=np.sin)
        probes = [nengo.Probe(u, sample_every=p, synapse=5*p) for p in periods]

    sim = Simulator(m, dt=dt)
    sim.run(0.333)
    for i, p in enumerate(periods):
        assert len(sim.trange(p)) == len(sim.data[probes[i]])


def test_signal_indexing_1(RefSimulator):
    one = Signal(np.zeros(1), name="a")
    two = Signal(np.zeros(2), name="b")
    three = Signal(np.zeros(3), name="c")
    tmp = Signal(np.zeros(3), name="tmp")

    m = Model(dt=0.001)
    m.operators += [
        Reset(one), Reset(two), Reset(tmp),
        DotInc(Signal(1, name="A1"), three[:1], one),
        DotInc(Signal(2.0, name="A2"), three[1:], two),
        DotInc(
            Signal([[0, 0, 1], [0, 1, 0], [1, 0, 0]], name="A3"), three, tmp),
        Copy(src=tmp, dst=three, as_update=True),
    ]

    sim = RefSimulator(None, model=m)
    sim.signals[three] = np.asarray([1, 2, 3])
    sim.step()
    assert np.all(sim.signals[one] == 1)
    assert np.all(sim.signals[two] == [4, 6])
    assert np.all(sim.signals[three] == [3, 2, 1])
    sim.step()
    assert np.all(sim.signals[one] == 3)
    assert np.all(sim.signals[two] == [4, 2])
    assert np.all(sim.signals[three] == [1, 2, 3])


def test_simple_pyfunc(RefSimulator):
    dt = 0.001
    time = Signal(np.zeros(1), name="time")
    sig = Signal(np.zeros(1), name="sig")
    m = Model(dt=dt)
    sig_in, sig_out = build_pyfunc(m, lambda t, x: np.sin(x), True, 1, 1, None)
    m.operators += [
        Reset(sig),
        DotInc(Signal([[1.0]]), time, sig_in),
        DotInc(Signal([[1.0]]), sig_out, sig),
        DotInc(Signal(dt), Signal(1), time, as_update=True),
    ]

    sim = RefSimulator(None, model=m)
    for i in range(5):
        sim.step()
        t = i * dt
        assert np.allclose(sim.signals[sig], np.sin(t))
        assert np.allclose(sim.signals[time], t + dt)


def test_probedict():
    """Tests simulator.ProbeDict's implementation."""
    raw = {"scalar": 5,
           "list": [2, 4, 6]}
    probedict = nengo.simulator.ProbeDict(raw)
    assert np.all(probedict["scalar"] == np.asarray(raw["scalar"]))
    assert np.all(probedict.get("list") == np.asarray(raw.get("list")))


def test_noise(RefSimulator, seed):
    """Make sure that we can generate noise properly."""

    n = 1000
    mean, std = 0.1, 0.8
    noise = Signal(np.zeros(n), name="noise")
    process = nengo.processes.StochasticProcess(
        nengo.dists.Gaussian(mean, std))

    m = Model(dt=0.001)
    m.operators += [Reset(noise), SimNoise(noise, process)]

    sim = RefSimulator(None, model=m, seed=seed)
    samples = np.zeros((100, n))
    for i in range(100):
        sim.step()
        samples[i] = sim.signals[noise]

    h, xedges = np.histogram(samples.flat, bins=51)
    x = 0.5 * (xedges[:-1] + xedges[1:])
    dx = np.diff(xedges)
    z = 1./np.sqrt(2 * np.pi * std**2) * np.exp(-0.5 * (x - mean)**2 / std**2)
    y = h / float(h.sum()) / dx
    assert np.allclose(y, z, atol=0.02)

    def test_real_time_model(self, tol=.001):

        import time

        m = nengo.Model( "test_real_time_model" )
        node = m.make_node( "node", [0] )
        ens = m.make_ensemble( "ens", nengo.LIF(35), dimensions=1 )
        m.connect( node, ens )

        sim = m.simulator( sim_class=self.Simulator, real_time=True )
        t_start = time.time()
        sim.run( 2 )
        t_end = time.time()
        self.assertTrue( abs( t_end - t_start - 2 ) < tol )


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, "-v"])
