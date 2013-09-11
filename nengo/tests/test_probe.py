import nengo
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestProbe(SimulatorTestCase):

    def test_long_name(self):
        m = nengo.Model('test_long_name', seed=123)
        m.make_ensemble(("This is an extremely long name that will test "
                         "if we can access sim data with long names"),
                        nengo.LIF(10), 1)
        m.probe("This is an extremely long name that will test "
                "if we can access sim data with long names")

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(0.01)

        self.assertIsNotNone(sim.data(
            "This is an extremely long name that will test "
            "if we can access sim data with long names"))

    unittest.skip('known to fail, and problematic as discussed in #97')
    def test_two_probes(self):

        m = nengo.Model('model')
        m.make_ensemble('A', nengo.LIF(10), 1)
        x = m.probe('A')
        y = m.probe('A')
        sim = m.simulator(sim_class=self.Simulator)
        sim.run(0.1)
        self.assertIsNotNone(sim.data('A'))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
