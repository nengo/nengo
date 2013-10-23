import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.builder import Direct, Signal, Constant
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

        tmp = m.add(Signal(n=3, name='tmp'))

        m._operators += [simulator.ProdUpdate(
            Constant(1), three[0:1], Constant(0), one)]
        m._operators += [simulator.ProdUpdate(
            Constant(2.0), three[1:], Constant(0), two)]
        m._operators += [
            simulator.Reset(tmp),
            simulator.DotInc(
                Constant([[0,0,1],[0,1,0],[1,0,0]]),
                three,
                tmp),
            simulator.Copy(src=tmp, dst=three, as_update=True),
            ]

        sim = m.simulator(sim_class=self.Simulator)
        memo = sim.model.memo
        sim.signals[sim.get(three)] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.get(one)] == 1))
        self.assertTrue(np.all(sim.signals[sim.get(two)] == [4, 6]))
        self.assertTrue(np.all(sim.signals[sim.get(three)] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.get(one)] == 3))
        self.assertTrue(np.all(sim.signals[sim.get(two)] == [4, 2]))
        self.assertTrue(np.all(sim.signals[sim.get(three)] == [1, 2, 3]))

    def test_simple_direct_mode(self):
        m = nengo.Model("test_simple_direct_mode")
        sig = m.add(Signal(n=1, name='sig'))

        pop = m.add(Direct(n_in=1, n_out=1, fn=np.sin))

        m._operators += [simulator.DotInc(Constant([[1.0]]),
                                          m.t, pop.input_signal)]
        m._operators += [simulator.ProdUpdate(
            Constant([[1.0]]), pop.output_signal, Constant(0), sig)]

        sim = m.simulator(sim_class=self.Simulator)
        #sim.print_op_groups()
        sim.step()
        #print 'after step:'
        #print sim.signals
        dt = sim.model.dt
        for i in range(5):
            sim.step()

            t = (i + 2) * dt
            self.assertTrue(np.allclose(sim.signals[sim.get(m.t)], t),
                msg='%s != %s' % (sim.signals[sim.get(m.t)], t))
            self.assertTrue(
                np.allclose(
                    sim.signals[sim.get(sig)], np.sin(t - dt*2)),
                msg='%s != %s' % (sim.signals[sim.get(sig)], np.sin(t - dt*2)))

    def test_encoder_decoder_pathway(self):
        #
        # This test is a very short and small simulation that
        # verifies (like by hand) that the simulator does the right
        # things in the right order.
        #
        m = nengo.Model("")
        foo = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))

        decoders = np.asarray([.2,.1])
        decs = Constant(decoders*0.5)
        m._operators.append(
            simulator.DotInc(
                Constant([[1.0],[2.0]]),
                foo,
                pop.input_signal,
                tag='dotinc'))
        m._operators.append(
            simulator.ProdUpdate(
                decs,
                pop.output_signal,
                Constant(0.2),
                foo,
                tag='produp'))

        sim = m.simulator(sim_class=self.Simulator)

        def check(sig, target):
            self.assertTrue(np.allclose(sim.signals[sim.get(sig)], target),
                            "%s: value %s is not close to target %s" %
                            (sig, sim.signals[sim.get(sig)], target))

        # -- initialize things
        sim.signals[sim.get(foo)] = np.asarray([1.0])
        check(foo, 1.0)
        check(pop.input_signal, 0)
        check(pop.output_signal, 0)

        sim.step()
        #DotInc to pop.input_signal (input=[1.0,2.0])
        #produpdate updates foo (foo=[0.2])
        #pop updates pop.output_signal (output=[2,3])

        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])
        check(foo, .2)
        check(decs, [.1, .05])

        sim.step()
        #DotInc to pop.input_signal (input=[0.2,0.4])
        # (note that pop resets its own input signal each timestep)
        #produpdate updates foo (foo=[0.39]) 0.2*0.5*2+0.1*0.5*3 + 0.2*0.2
        #pop updates pop.output_signal (output=[1.2,1.4])

        check(decs, [.1, .05])
        check(pop.input_signal, [0.2, 0.4])
        check(pop.output_signal, [1.2, 1.4])
        # -- foo is computed as a prodUpdate of the *previous* output signal
        #    foo <- .2 * foo + dot(decoders * .5, output_signal)
        #           .2 * .2  + dot([.2, .1] * .5, [2, 3])
        #           .04      + (.2 + .15)
        #        <- .39
        check(foo, .39)

    def test_encoder_decoder_with_views(self):
        m = nengo.Model("")
        foo = m.add(Signal(n=1, name='foo'))
        pop = m.add(Direct(n_in=2, n_out=2, fn=lambda x: x + 1, name='pop'))

        decoders = np.asarray([.2,.1])
        m._operators += [simulator.DotInc(Constant([[1.0],[2.0]]),
                                          foo[:], pop.input_signal)]
        m._operators += [simulator.ProdUpdate(
            Constant(decoders*0.5),pop.output_signal, Constant(0.2), foo[:])]

        sim = m.simulator(sim_class=self.Simulator)

        def check(sig, target):
            self.assertTrue(np.allclose(sim.signals[sim.get(sig)], target),
                            "%s: value %s is not close to target %s" %
                            (sig, sim.signals[sim.get(sig)], target))

        #set initial value of foo (foo=1.0)
        sim.signals[sim.get(foo)] = np.asarray([1.0])
        #pop.input_signal = [0,0]
        #pop.output_signal = [0,0]

        sim.step()
        #DotInc to pop.input_signal (input=[1.0,2.0])
        #produpdate updates foo (foo=[0.2])
        #pop updates pop.output_signal (output=[2,3])

        check(foo, .2)
        check(pop.input_signal, [1, 2])
        check(pop.output_signal, [2, 3])

        sim.step()
        #DotInc to pop.input_signal (input=[0.2,0.4])
        # (note that pop resets its own input signal each timestep)
        #produpdate updates foo (foo=[0.39]) 0.2*0.5*2+0.1*0.5*3 + 0.2*0.2
        #pop updates pop.output_signal (output=[1.2,1.4])

        check(foo, .39)
        check(pop.input_signal, [0.2, 0.4])
        check(pop.output_signal, [1.2, 1.4])

if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
