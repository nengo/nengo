import logging
import time

import numpy as np

import nengo
from nengo.tests.helpers import SimulatorTestCase, unittest, assert_allclose


logger = logging.getLogger(__name__)


class TestProbe(SimulatorTestCase):

    def test_multirun(self):
        """Test probing the time on multiple runs"""
        rng = np.random.RandomState(2239)

        # set rtol a bit higher, since OCL model.t accumulates error over time
        rtol = 1e-4

        model = nengo.Model("Multi-run")

        sim = self.Simulator(model)
        dt = sim.model.dt

        # t_stops = [0.123, 0.283, 0.821, 0.921]
        t_stops = dt * rng.randint(low=100, high=2000, size=10)

        t_sum = 0
        for ti in t_stops:
            sim.run(ti)
            sim_t = sim.time()
            t = dt * np.arange(len(sim_t))
            self.assertTrue(np.allclose(sim_t, t, rtol=rtol))
            # assert_allclose(self, logger, sim_t, t, rtol=rtol)

            t_sum += ti
            self.assertTrue(np.allclose(sim_t[-1], t_sum - dt, rtol=rtol))

    def test_dts(self):
        """Test probes with different sampling times."""

        n = 10

        rng = np.random.RandomState(48392)
        dts = 0.001 * rng.randint(low=1, high=100, size=n)
        # dts = 0.001 * np.hstack([2, rng.randint(low=1, high=100, size=n-1)])

        def input_fn(t):
            return range(1, 10)

        model = nengo.Model('test_probe_dts', seed=2891)
        with model:
            probes = []
            for i, dt in enumerate(dts):
                xi = nengo.Node(label='x%d' % i, output=input_fn)
                p = nengo.Probe(xi, 'output', sample_every=dt)
                probes.append(p)

        sim = self.Simulator(model)
        simtime = 2.483
        # simtime = 2.484
        dt = sim.model.dt

        timer = time.time()
        sim.run(simtime)
        timer = time.time() - timer
        logger.debug(
            "Ran %(n)s probes for %(simtime)s sec simtime in %(timer)0.3f sec"
            % locals())

        for i, p in enumerate(probes):
            t = dt * np.arange(int(np.ceil(simtime / dts[i])))
            x = np.asarray(map(input_fn, t))
            y = sim.data(p)
            self.assertTrue(len(x) == len(y))
            self.assertTrue(np.allclose(y[1:], x[:-1]))  # 1-step delay

    def test_large(self):
        """Test with a lot of big probes. Can also be used for speed."""

        n = 10

        # rng = np.random.RandomState(48392)
        # input_val = rng.normal(size=100).tolist()
        def input_fn(t):
            return range(1, 10)
            # return input_val
            # return [np.sin(t), np.cos(t)]

        model = nengo.Model('test_large_probes', seed=3249)

        with model:
            probes = []
            for i in xrange(n):
                xi = nengo.Node(label='x%d' % i, output=input_fn)
                probes.append(nengo.Probe(xi, 'output'))
                # Ai = m.make_ensemble('A%d' % i, nengo.LIF(n_neurons), 1)
                # m.connect(xi, Ai)
                # m.probe(Ai, filter=0.1)

        sim = self.Simulator(model)
        simtime = 2.483
        dt = sim.model.dt

        timer = time.time()
        sim.run(simtime)
        timer = time.time() - timer
        logger.debug(
            "Ran %(n)s probes for %(simtime)s sec simtime in %(timer)0.3f sec"
            % locals())

        t = dt * np.arange(int(np.round(simtime / dt)))
        x = np.asarray(map(input_fn, t))
        for p in probes:
            y = sim.data(p)
            self.assertTrue(np.allclose(y[1:], x[:-1]))  # 1-step delay


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
