import nengo

from nengo.tests.helpers import SimulatorTestCase, unittest

class TestContext(SimulatorTestCase):
    def test_default(self):
        model = nengo.Model("test")

        e = nengo.Node(nengo.LIF(1), 1)
        n = nengo.Node([0])

        self.assertTrue(e in model.objs)
        self.assertTrue(n in model.objs)

        con = MyContext()
        with con:
            e2 = nengo.Ensemble(nengo.LIF(1), 1)

        self.assertTrue(e2 in con.objs)
        self.assertTrue(not e2 in model.objs)

        e3 = nengo.Ensemble(nengo.LIF(1), 1)

        self.assertTrue(e3 in model.objs)

        model2 = nengo.Model("test")

        e4 = nengo.Ensemble(nengo.LIF(1), 1)

        self.assertTrue(not e4 in model.objs)
        self.assertTrue(e4 in model2.objs)

    def test_with(self):
        con1 = MyContext()
        con2 = MyContext()
        con3 = MyContext()

        with con1:
            e1 = nengo.Ensemble(nengo.LIF(1), 1)
            self.assertTrue(e1 in con1.objs)

            with con2:
                e2 = nengo.Ensemble(nengo.LIF(1), 1)
                self.assertTrue(e2 in con2.objs)
                self.assertTrue(not e2 in con1.objs)

                with con3:
                    e3 = nengo.Ensemble(nengo.LIF(1), 1)
                    self.assertTrue(e3 in con3.objs)
                    self.assertTrue(not e3 in con2.objs and not e3 in con1.objs)

                e4 = nengo.Ensemble(nengo.LIF(1), 1)
                self.assertTrue(e4 in con2.objs)
                self.assertTrue(not e4 in con3.objs)

            e5 = nengo.Ensemble(nengo.LIF(1), 1)
            self.assertTrue(e5 in con1.objs)

        e6 = nengo.Ensemble(nengo.LIF(1), 1)
        self.assertTrue(not e6 in con1.objs)

    def test_networks(self):
        ###TODO
        pass


class MyContext():
    def __init__(self):
        self.objs = []

    def add(self, obj):
        self.objs += [obj]

    def __enter__(self):
        nengo.context.append(self)

    def __exit__(self, exception_type, exception_value, traceback):
        nengo.context.pop()


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
