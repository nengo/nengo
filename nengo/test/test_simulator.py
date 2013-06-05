import numpy as np
from nengo.simulator_objects import SimModel
from nengo.nonlinear import Direct
from nengo.simulator import Simulator


def test_signal_indexing_1():
    m = SimModel()
    one = m.signal(1)
    two = m.signal(2)
    three = m.signal(3)

    m.filter(1, three[0], one)
    m.filter(2.0, three[1:], two)
    m.filter([[0, 0, 1], [0, 1, 0], [1, 0, 0]], three, three)

    sim = Simulator(m)
    sim.signals[three] = np.asarray([1, 2, 3])
    sim.step()
    assert np.all(sim.signals[one] == 1)
    assert np.all(sim.signals[two] == [4, 6])
    assert np.all(sim.signals[three] == [3, 2, 1])
    sim.step()
    assert np.all(sim.signals[one] == 3)
    assert np.all(sim.signals[two] == [4, 2])
    assert np.all(sim.signals[three] == [1, 2, 3])


def setup_simtime(m):
    steps = m.signal()
    simtime = m.signal()
    one = m.signal(value=1.0)

    # -- steps counts by 1.0
    m.filter(1.0, steps, steps)
    m.filter(1.0, one, steps)

    # simtime <- dt * steps
    m.filter(m.dt, steps, simtime)
    m.filter(m.dt, one, simtime)

    return one, steps, simtime


def test_simple_direct_mode():
    m = SimModel()
    one, steps, simtime = setup_simtime(m)
    sig = m.signal()

    pop = m.nonlinearity(
        Direct(n_in=1, n_out=1, fn=np.sin))
    m.encoder(simtime, pop, weights=[[1.0]])
    m.decoder(pop, sig, weights=[[1.0]])
    m.transform(1.0, sig, sig)

    sim = Simulator(m)
    for i in range(5):
        sim.step()
        if i:
            assert sim.signals[sig] == np.sin(sim.signals[simtime] - .001)


