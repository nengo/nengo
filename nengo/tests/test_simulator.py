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

    def test_encoder_decoder_pathway(self):
        m = nengo.Model("")
        one = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))
        enc = m.add(Encoder(one, pop, [[1.0], [2.0]]))
        dec = m.add(Decoder(pop, one, [[.2, .1]]))
        tf = m.add(Transform(0.5, one, one))
        fi = m.add(Filter(0.2, one, one))

        sim = self.Simulator(m)
        sim.signals[one] = np.asarray([1.0])
        def pp(sig, target):
            print sig, sim.signals[sig], target
        sim.step()
        pp(one, .55)
        pp(enc.sig, .55) # -- was 1.0 during step fn
        pp(enc.weights_signal, [[1], [2]]) #
        pp(pop.input_signal, [1, 2])
        pp(pop.output_signal, [2, 3])
        #pp(sim.dec_outputs[dec.sig], [.7])

        self.assertTrue(np.allclose(sim.signals[one], .55, atol=.01, rtol=.01),
                        msg=str(sim.signals[one]))

    def test_encoder_decoder_with_views(self):
        m = nengo.Model("")
        one = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))
        enc = m.add(Encoder(one[:], pop, [[1.0], [2.0]]))
        dec = m.add(Decoder(pop, one, [[.2, .1]]))
        tf = m.add(Transform(0.5, one, one[:]))
        fi = m.add(Filter(0.2, one[:], one))

        sim = self.Simulator(m)
        sim.signals[one] = np.asarray([1.0])
        def pp(sig, target):
            #print sig, sim.signals[sig], target
            pass
        sim.step()
        pp(one, .55)
        pp(enc.sig, .55) # -- was 1.0 during step fn
        pp(enc.weights_signal, [[1], [2]]) #
        pp(pop.input_signal, [1, 2])
        pp(pop.output_signal, [2, 3])
        #pp(sim.dec_outputs[dec.sig], [.7])

        self.assertTrue(np.allclose(sim.signals[one], .55, atol=.01, rtol=.01),
                        msg=sim.signals[one])



if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
