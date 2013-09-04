import numpy as np

import nengo
from nengo.objects import Direct, Encoder, Decoder, Filter, Signal, Transform
from nengo.tests.helpers import SimulatorTestCase, unittest

import logging
logger = logging.getLogger(__name__)

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
        sim.step()
        for i in range(5):
            sim.step()
            t = (i + 2) * m.dt
            self.assertTrue(np.allclose(sim.signals[m.simtime], t))
            self.assertTrue(np.allclose(sim.signals[sig], np.sin(t - m.dt)))

    def test_encoder_decoder_pathway(self):
        m = nengo.Model("")
        foo = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))
        enc = m.add(Encoder(foo, pop, [[1.0], [2.0]]))
        dec = m.add(Decoder(pop, foo, [[.2, .1]]))
        tf = m.add(Transform(0.5, foo, foo))
        fi = m.add(Filter(0.2, foo, foo))

        sim = self.Simulator(m)
        sim.signals[foo] = np.asarray([1.0])
        sim.step()

        def check(sig, target):
            if not np.allclose(sim.signals[sig], target):
                msg = ("%s: value %s is not close to target %s" %
                       (sig, sim.signals[sig], target))
                raise AssertionError(msg)
        check(foo, .55)
        check(enc.sig, .55) # -- was 1.0 during step fn
        check(enc.weights_signal, [[1], [2]]) #
        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])
        check(sim.dec_outputs[dec.sig], [.7])

        self.assertTrue(np.allclose(sim.signals[foo], .55, atol=.01, rtol=.01),
                        msg=str(sim.signals[foo]))

    def test_encoder_decoder_with_views(self):
        m = nengo.Model("")
        foo = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))
        enc = m.add(Encoder(foo[:], pop, [[1.0], [2.0]]))
        dec = m.add(Decoder(pop, foo, [[.2, .1]]))
        tf = m.add(Transform(0.5, foo, foo[:]))
        fi = m.add(Filter(0.2, foo[:], foo))

        sim = self.Simulator(m)
        sim.signals[foo] = np.asarray([1.0])
        sim.step()

        def check(sig, target):
            if not np.allclose(sim.signals[sig], target):
                msg = ("%s: value %s is not close to target %s" %
                       (sig, sim.signals[sig], target))
                raise AssertionError(msg)
        check(foo, .55)
        check(enc.sig, .55) # -- was 1.0 during step fn
        check(enc.weights_signal, [[1], [2]]) #
        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])
        check(sim.dec_outputs[dec.sig], [.7])

        self.assertTrue(np.allclose(sim.signals[foo], .55, atol=.01, rtol=.01),
                        msg=sim.signals[foo])



if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
