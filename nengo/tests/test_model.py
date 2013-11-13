import numpy as np

import nengo
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestModelBuild(unittest.TestCase):
    def test_build(self):
        m = nengo.Model('test_build', seed=123)
        with m:
            input = nengo.Node(output=1)
            A = nengo.Ensemble(nengo.LIF(40), 1)
            B = nengo.Ensemble(nengo.LIF(20), 1)
            nengo.Connection(input, A)
            nengo.DecodedConnection(A, B, function=lambda x: x ** 2)
            input_p = nengo.Probe(input, 'output')
            A_p = nengo.Probe(A, 'decoded_output', filter=0.01)
            B_p = nengo.Probe(B, 'decoded_output', filter=0.01)

        mcopy = nengo.Simulator(m, dt=0.001).model
        self.assertItemsEqual([o.label for o in m.objs], [o.label for o in mcopy.objs])

        def compare_objs(orig, copy, attrs):
            for attr in attrs:
                self.assertEqual(getattr(orig, attr), getattr(copy, attr))
            for attr in ('connections_in', 'connections_out'):
                self.assertEqual(len(getattr(orig, attr)),
                                 len(getattr(copy, attr)))
                for o_c, c_c in zip(getattr(orig, attr), getattr(copy, attr)):
                    compare_connections(o_c, c_c)
            for p_o, p_c in zip(orig.probes.values(), copy.probes.values()):
                self.assertEqual(len(p_o), len(p_c))

        def compare_connections(orig, copy):
            self.assertEqual(orig.filter, copy.filter)
            self.assertEqual(orig.transform, copy.transform)
            for p_o, p_c in zip(orig.probes.values(), copy.probes.values()):
                self.assertEqual(len(p_o), len(p_c))
        
        
        ens_attrs = ('label', 'dimensions', 'radius')
        
        for o,copy_o in zip(m.objs,mcopy.objs):
            if isinstance(o, nengo.Node):
                compare_objs(o, copy_o, ('output',))
            else:
                compare_objs(o, copy_o, ens_attrs)
                compare_objs(o, copy_o, ens_attrs)

    def test_seeding(self):
        """Test that setting the model seed fixes everything"""

        ### TODO: this really just checks random parameters in ensembles.
        ###   Are there other objects with random parameters that should be
        ###   tested? (Perhaps initial weights of learned connections)

        m = nengo.Model('test_seeding')
        with m:
            input = nengo.Node(output=1, label='input')
            A = nengo.Ensemble(nengo.LIF(40), 1, label='A')
            B = nengo.Ensemble(nengo.LIF(20), 1, label='B')
            nengo.Connection(input, A)
            nengo.DecodedConnection(A, B, function=lambda x: x ** 2)
            input_p = nengo.Probe(input, 'output')
            A_p = nengo.Probe(A, 'decoded_output', filter=0.01)
            B_p = nengo.Probe(B, 'decoded_output', filter=0.01)

        m.seed = 872
        m1 = nengo.Simulator(m, dt=0.001).model
        m2 = nengo.Simulator(m, dt=0.001).model
        m.seed = 873
        m3 = nengo.Simulator(m, dt=0.001).model

        def compare_objs(obj1, obj2, attrs, equal=True):
            for attr in attrs:
                check = (np.all(getattr(obj1, attr) == getattr(obj2, attr))
                         if equal else
                         np.any(getattr(obj1, attr) != getattr(obj2, attr)))
                if not check:
                    print getattr(obj1, attr)
                    print getattr(obj2, attr)
                self.assertTrue(check)

        ens_attrs = ('encoders', 'max_rates', 'intercepts')
        A = [[o for o in mi.objs if o.label == 'A'][0] for mi in [m1, m2, m3]]
        B = [[o for o in mi.objs if o.label == 'B'][0] for mi in [m1, m2, m3]]
        compare_objs(A[0], A[1], ens_attrs)
        compare_objs(B[0], B[1], ens_attrs)
        compare_objs(A[0], A[2], ens_attrs, equal=False)
        compare_objs(B[0], B[2], ens_attrs, equal=False)

        neur_attrs = ('gain', 'bias')
        compare_objs(A[0].neurons, A[1].neurons, neur_attrs)
        compare_objs(B[0].neurons, B[1].neurons, neur_attrs)
        compare_objs(A[0].neurons, A[2].neurons, neur_attrs, equal=False)
        compare_objs(B[0].neurons, B[2].neurons, neur_attrs, equal=False)


class TestModel(SimulatorTestCase):

    def test_counters(self):
        m = nengo.Model('test_counters', seed=123)
        with m:
            p = nengo.Probe(m.steps, 'output')
        
        sim = self.Simulator(m, dt=0.001)
        sim.run(0.003)
        self.assertTrue(np.allclose(sim.data(m.t_probe).flatten(),
                                    [0.00, .001, .002]))
        self.assertTrue(np.allclose(sim.data(p).flatten(), [0, 1, 2]))


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
