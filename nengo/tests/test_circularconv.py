import numpy as np

import nengo
from nengo.core import Constant
from nengo.templates import EnsembleArray
from nengo.templates.circularconv import \
    DirectCircularConvolution, CircularConvolution
from nengo.tests.helpers import SimulatorTestCase, unittest, assert_allclose

import logging
logger = logging.getLogger(__name__)

class TestCircularConv(SimulatorTestCase):
    def test_helpers(self):
        """Test the circular convolution helper functions in Numpy"""
        from nengo.templates.circularconv import \
            _input_transform, _output_transform

        rng = np.random.RandomState(43232)

        dims = 1000
        invert_a = True
        invert_b = False
        x = rng.randn(dims)
        y = rng.randn(dims)
        z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

        dims2 = 2*dims - (2 if dims % 2 == 0 else 1)
        inA = _input_transform(dims, first=True, invert=invert_a)
        inB = _input_transform(dims, first=False, invert=invert_b)
        outC = _output_transform(dims)

        XY = np.zeros((dims2,2))
        XY += np.dot(inA, x)
        XY += np.dot(inB, y)

        C = XY[:,0] * XY[:,1]
        z1 = np.dot(outC, C)

        assert_allclose(self, logger, z0, z1)

    def test_direct(self):
        """Test direct-mode circular convolution"""

        dims = 1000

        rng = np.random.RandomState(8238)
        a = rng.randn(dims)
        b = rng.randn(dims)
        c = circconv(a, b)

        m = nengo.Model("")
        A = m.add(Constant(n=dims, value=a))
        B = m.add(Constant(n=dims, value=b))
        C = m.add(Signal(n=dims, name="C"))

        DirectCircularConvolution(m, A, B, C)

        sim = self.Simulator(m)
        sim.run_steps(10)
        c2 = sim.signals[C]

        # assert_allclose(self, logger, c, c2, atol=1e-5, rtol=1e-5)
        assert_allclose(self, logger, c, c2, atol=1e-5, rtol=1e-3)

    def test_circularconv(self, dims=5, neurons_per_product=100):
        rng = np.random.RandomState(42342)

        n_neurons = neurons_per_product * dims
        n_neurons_d = 2 * neurons_per_product * (
            2*dims - (2 if dims % 2 == 0 else 1))
        radius = 1

        model = nengo.Model("circular convolution")

        a = rng.normal(scale=np.sqrt(1./dims), size=dims)
        b = rng.normal(scale=np.sqrt(1./dims), size=dims)
        c = nengo.templates.circularconv.circconv(a, b)

        inputA = model.make_node("inputA", output=a)
        inputB = model.make_node("inputB", output=b)

        A = model.add(EnsembleArray('A', nengo.LIF(n_neurons), dims, radius=radius))
        B = model.add(EnsembleArray('B', nengo.LIF(n_neurons), dims, radius=radius))
        C = model.add(EnsembleArray('C', nengo.LIF(n_neurons), dims, radius=radius))

        inputA.connect_to(A)
        inputB.connect_to(B)

        D = CircularConvolution(model, A, B, C, neurons=nengo.LIF(n_neurons_d),
                                radius=radius, name="D", pstc=0.005)

        model.probe(A, filter=0.03)
        model.probe(B, filter=0.03)
        model.probe(C, filter=0.03)
        model.probe(D.ensemble, filter=0.03)

        sim = model.simulator()
        sim.run(1.0)

        t = sim.data(model.t).flatten()
        a_sim = sim.data(A)[t > 0.5].mean(0)
        b_sim = sim.data(B)[t > 0.5].mean(0)
        c_sim = sim.data(C)[t > 0.5].mean(0)
        d_sim = sim.data(D.ensemble)[t > 0.5].mean(0)

        d = np.dot(D.transform_inA, a) + np.dot(D.transform_inB, b)
        m = np.prod(d.reshape((-1,2)), axis=1)

        # print "max a", np.abs(a).max()
        # print "max b", np.abs(b).max()
        # print "max c", np.abs(c).max()
        print "max multiply", np.abs(m).max()

        print c
        print c_sim

        rtol, atol = 0.05, 0.01
        self.assertTrue(np.allclose(a, a_sim, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(b, b_sim, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(d, d_sim, rtol=rtol, atol=atol))
        rtol, atol = 0.1, 0.05
        self.assertTrue(np.allclose(c, c_sim, rtol=rtol, atol=atol))

        # import matplotlib.pyplot as plt
        # def plot(sim, a, A, title=""):
        #     plt.figure()
        #     t = sim.data(model.t)
        #     a_ref = np.tile(a, (len(t), 1))
        #     a_sim = sim.data(A)
        #     print a_sim.shape
        #     colors = ['b', 'g', 'r', 'c', 'm', 'y']
        #     for i in xrange(dims):
        #         plt.plot(t, a_ref[:,i], '--', color=colors[i])
        #         plt.plot(t, a_sim[:,i], '-', color=colors[i])
        #     plt.title(title)

        # plot(sim, a, A, title="A")
        # plot(sim, b, B, title="B")
        # plot(sim, c, C, title="C")
        # plot(sim, d, D.ensemble, title="D")
        # plt.show()


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
