import numpy as np
from nengo.simulator_objects import SimModel
from nengo.simulator import Simulator


def test_signal_indexing_1():
    m = SimModel()
    one = m.signal(1)
    two = m.signal(2)
    three = m.signal(3, value=[1, 2, 3])

    m.filter(1, three[0], one)
    m.filter(2.0, three[1:], two)
    m.filter([[0, 0, 1], [0, 1, 0], [1, 0, 0]], three, three)

    sim = Simulator(m)
    sim.step()
    assert np.all(sim.signals[one] == 1)
    assert np.all(sim.signals[two] == [4, 6])
    assert np.all(sim.signals[three] == [3, 2, 1])
    sim.step()
    assert np.all(sim.signals[one] == 3)
    assert np.all(sim.signals[two] == [4, 2])
    assert np.all(sim.signals[three] == [1, 2, 3])
