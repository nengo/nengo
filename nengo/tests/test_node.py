import numpy as np

import nengo
import nengo.old_api as nef
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestNode(SimulatorTestCase):

    def test_simple(self):
        params = dict(simulator=self.Simulator, seed=123, dt=0.001)

        # Old API
        net = nef.Network('test_simple', **params)
        net.make_input('in', value=np.sin)
        net.make_probe('in', dt_sample=0.001, pstc=0.0)
        net.run(0.01)
        self.assertTrue(np.allclose(net.model.data[net.model.t].ravel(),
                                    np.arange(0.001, 0.0105, .001)))
        self.assertTrue(np.allclose(net.model.data['in'].ravel(),
                                    np.sin(np.arange(0, 0.0095, .001))))

        # New API
        m = nengo.Model('test_simple', **params)
        node = m.make_node('in', output=np.sin)
        m.probe('in')
        m.run(0.01)
        self.assertTrue(np.allclose(m.data[m.t].ravel(),
                                    np.arange(0.001, 0.0105, .001)))
        self.assertTrue(np.allclose(m.data['in'].ravel(),
                                    np.sin(np.arange(0, 0.0095, .001))))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
