import numpy as np

import nengo
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestBasalGanglia(SimulatorTestCase):
    def test_basic(self):
        model = nengo.Model('test_basalganglia_basic')

        with model:
            bg = nengo.networks.BasalGanglia(dimensions=5, label='BG')
            input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label='input')
            nengo.Connection(input, bg.input)
            p = nengo.Probe(bg.output, 'output')

        sim = self.Simulator(model)
        sim.run(0.2)

        output = np.mean(sim.data(p)[50:], axis=0)

        self.assertGreater(output[0], -0.1)
        self.assertLess(output[1], -0.8)
        self.assertLess(output[2], -0.8)
        self.assertLess(output[3], -0.8)
        self.assertLess(output[4], -0.8)


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
