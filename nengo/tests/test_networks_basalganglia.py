import numpy as np

import nengo
from nengo.networks.basalganglia import BasalGanglia
from nengo.templates import EnsembleArray
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestBasalGanglia(SimulatorTestCase):
    def test_basic(self):

        model = nengo.Model('test_basalganglia_basic')
        
        model.add(BasalGanglia('BG', dimensions=5))

        model.make_node('input', [0.8, 0.4, 0.4, 0.4, 0.4])

        model.connect('input', 'BG.input')

        model.probe('BG.output')
        
        sim = model.simulator(sim_class=self.Simulator)
        
        sim.run(0.2)
        
        output = np.mean(sim.data('BG.output')[50:], axis=0)
        
        self.assertGreater(output[0], -0.1)
        self.assertLess(output[1], -0.8)
        self.assertLess(output[2], -0.8)
        self.assertLess(output[3], -0.8)
        self.assertLess(output[4], -0.8)
        
        


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
