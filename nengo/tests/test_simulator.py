import numpy as np

import nengo
import nengo.simulator as simulator
import nengo.core as core
from nengo.core import Direct, Signal, Constant
from nengo.tests.helpers import SimulatorTestCase, unittest

import logging
logger = logging.getLogger(__name__)

class TestSimulator(SimulatorTestCase):

    def test_steps(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=self.Simulator)
        self.assertEqual(0, sim.signals[m.steps])
        sim.step()
        self.assertEqual(1, sim.signals[m.steps])
        sim.step()
        self.assertEqual(2, sim.signals[m.steps])

    def test_time(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=self.Simulator)
        self.assertEqual(0.00, sim.signals[m.t])
        sim.step()
        self.assertEqual(0.001, sim.signals[m.t])
        sim.step()
        self.assertEqual(0.002, sim.signals[m.t])

    def test_signal_indexing_1(self):
        m = nengo.Model("test_signal_indexing_1")
        one = m.add(Signal(n=1, name='a'))
        two = m.add(Signal(n=2, name='b'))
        three = m.add(Signal(n=3, name='c'))

        m._operators += [simulator.ProdUpdate(core.Constant(1), three[0:1], core.Constant(0), one)]
        m._operators += [simulator.ProdUpdate(core.Constant(2.0), three[1:], core.Constant(0), two)]
        m._operators += [simulator.DotInc(core.Constant([[0,0,1], [0,1,0], [1,0,0]]), three, m._get_output_view(three))]

        sim = m.simulator(sim_class=self.Simulator)
        memo = sim.model.memo
        sim.signals[sim.copied(three)] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.copied(one)] == 1))
        self.assertTrue(np.all(sim.signals[sim.copied(two)] == [4, 6]))
        self.assertTrue(np.all(sim.signals[sim.copied(three)] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.copied(one)] == 3))
        self.assertTrue(np.all(sim.signals[sim.copied(two)] == [4, 2]))
        self.assertTrue(np.all(sim.signals[sim.copied(three)] == [1, 2, 3]))

    def test_simple_direct_mode(self):
        m = nengo.Model("test_simple_direct_mode")
        sig = m.add(Signal(n=1, name='sig'))

        pop = m.add(Direct(n_in=1, n_out=1, fn=np.sin))
        
        m._operators += [simulator.DotInc(core.Constant([[1.0]]), m.t, pop.input_signal)]
        m._operators += [simulator.ProdUpdate(core.Constant([[1.0]]), pop.output_signal, core.Constant(0), sig)]

        sim = m.simulator(sim_class=self.Simulator)
        #sim.print_op_groups()
        sim.step()
        #print 'after step:'
        #print sim.signals
        dt = sim.model.dt
        for i in range(5):
            sim.step()

            t = (i + 2) * dt
            self.assertTrue(np.allclose(sim.signals[sim.copied(m.t)], t),
                msg='%s != %s' % (sim.signals[sim.copied(m.t)], t))
            self.assertTrue(
                np.allclose(
                    sim.signals[sim.copied(sig)], np.sin(t - dt)),
                msg='%s != %s' % (sim.signals[sim.copied(sig)], np.sin(t - dt)))

    def test_encoder_decoder_pathway(self):
        m = nengo.Model("")
        foo = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))
        
        decoders = np.asarray([.2,.1])
        m._operators += [simulator.DotInc(Constant([[1.0],[2.0]]), foo, pop.input_signal)]
        m._operators += [simulator.ProdUpdate(Constant(decoders*0.5), pop.output_signal, Constant(0.2), foo)]

        sim = m.simulator(sim_class=self.Simulator)
        sim.signals[sim.copied(foo)] = np.asarray([1.0])
        sim.step()

        def check(sig, target):
            self.assertTrue(np.allclose(sim.signals[sim.copied(sig)], target),
                            "%s: value %s is not close to target %s" %
                            (sig, sim.signals[sim.copied(sig)], target))
        check(foo, .55)
        try:
            check(pop.input_signal, [1, 2])
            check(pop.output_signal, [2, 3])
        except AssertionError:
            # -- not passing in reference simulator because
            #    input signals are zerod
            check(pop.input_signal, [0, 0])
            check(pop.output_signal, [0, 0])


        self.assertTrue(np.allclose(sim.signals[sim.copied(foo)],
                                    .55, atol=.01, rtol=.01),
                        msg=str(sim.signals[sim.copied(foo)]))

    def test_encoder_decoder_with_views(self):
        m = nengo.Model("")
        foo = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))

        decoders = np.asarray([.2,.1])
        m._operators += [simulator.DotInc(Constant([[1.0],[2.0]]), foo[:], pop.input_signal)]
        m._operators += [simulator.ProdUpdate(Constant(decoders*0.5), pop.output_signal, Constant(0.2), foo[:])]

        sim = m.simulator(sim_class=self.Simulator)
        sim.signals[sim.copied(foo)] = np.asarray([1.0])
        sim.step()

        def check(sig, target):
            self.assertTrue(np.allclose(sim.signals[sim.copied(sig)], target),
                            "%s: value %s is not close to target %s" %
                            (sig, sim.signals[sim.copied(sig)], target))
        check(foo, .55)
        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])

        self.assertTrue(np.allclose(sim.signals[sim.copied(foo)],
                                    .55, atol=.01, rtol=.01),
                        msg=sim.signals[sim.copied(foo)])


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
