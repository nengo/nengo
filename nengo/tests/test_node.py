try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
import nengo.old_api as nef

from helpers import SimulatorTestCase


class TestNode(SimulatorTestCase):

    def test_simple(self):
        params = dict(simulator=self.Simulator, seed=123, dt=0.001)

        # Old API
        net = nef.Network('test_simple', **params)
        net.make_input('in', value=np.sin)
        p = net.make_probe('in', dt_sample=0.001, pstc=0.0)
        rawp = net._raw_probe(net.inputs['in'], dt_sample=.001)
        st_probe = net._raw_probe(net.model.simtime, dt_sample=.001)
        net.run(0.01)

        data = p.get_data()
        raw_data = rawp.get_data()
        st_data = st_probe.get_data()
        assert np.allclose(st_data.ravel(),
                           np.arange(0.001, 0.0105, .001))
        assert np.allclose(raw_data.ravel(),
                           np.sin(np.arange(0, 0.0095, .001)))
        # -- the make_probe call induces a one-step delay
        #    on readout even when the pstc is really small.
        # TWB: But should it?
        assert np.allclose(data.ravel()[1:],
                           np.sin(np.arange(0, 0.0085, .001)))

        # New API
        m = nengo.Model('test_simple', **params)
        node = m.make_node('in', output=np.sin)
        m.probe('in')
        m.run(0.01)
        assert np.allclose(m.data[m.simtime].ravel(),
                           np.arange(0.001, 0.0105, .001))
        assert np.allclose(m.data['in'].ravel(),
                           np.sin(np.arange(0, 0.0095, .001)))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
