import numpy as np

import nengo
from nengo.objects import Direct, Encoder, Decoder, Filter, Signal, Transform
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestSimulator(SimulatorTestCase):

    def test_signal_indexing_1(self):
        m = nengo.Model("test_signal_indexing_1")
        one = m.add(Signal(n=1))
        two = m.add(Signal(n=2))
        three = m.add(Signal(n=3))

        m.add(Filter(1, three[0:1], one))
        m.add(Filter(2.0, three[1:], two))
        m.add(Filter([[0, 0, 1], [0, 1, 0], [1, 0, 0]], three, three))

        sim = self.Simulator(m)
        sim.signals[three] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 1))
        self.assertTrue(np.all(sim.signals[two] == [4, 6]))
        self.assertTrue(np.all(sim.signals[three] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 3))
        self.assertTrue(np.all(sim.signals[two] == [4, 2]))
        self.assertTrue(np.all(sim.signals[three] == [1, 2, 3]))

    def test_simple_direct_mode(self):
        m = nengo.Model("test_simple_direct_mode")
        sig = m.add(Signal())

        pop = m.add(Direct(n_in=1, n_out=1, fn=np.sin))
        m.add(Encoder(m.simtime, pop, weights=[[1.0]]))
        m.add(Decoder(pop, sig, weights=[[1.0]]))
        m.add(Transform(1.0, sig, sig))

        sim = self.Simulator(m)
        for i in range(5):
            sim.step()
            if i > 0:
                self.assertEqual(sim.signals[sig],
                                 np.sin(sim.signals[m.simtime] - .001))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
