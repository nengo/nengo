import pytest

import nengo


class MyContext():
    def __init__(self, offset=0):
        self.objs = {}
        self.offset = offset

    def add(self, obj):
        key = len(self.objs) + self.offset
        self.objs[key] = obj
        return key

    def __enter__(self):
        nengo.context.append(self)

    def __exit__(self, exception_type, exception_value, traceback):
        nengo.context.pop()


def test_default(Simulator):
    model = nengo.Model("test")

    e = nengo.Ensemble(nengo.LIF(1), 1)
    n = nengo.Node([0])
    assert e in model.objs.values()
    assert n in model.objs.values()

    con = MyContext()
    with con:
        e2 = nengo.Ensemble(nengo.LIF(1), 1)
    assert e2 in con.objs.values()
    assert not e2 in model.objs.values()

    e3 = nengo.Ensemble(nengo.LIF(1), 1)
    assert e3 in model.objs.values()

    model2 = nengo.Model("test2")
    e4 = nengo.Ensemble(nengo.LIF(1), 1)
    assert not e4 in model.objs.values()
    assert e4 in model2.objs.values()


def test_with(Simulator):
    model = nengo.Model('default')
    assert nengo.context[-1] == model
    con1 = MyContext()
    con2 = MyContext(offset=20)
    con3 = MyContext(offset=40)

    with con1:
        e1 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e1 in con1.objs.values()

        with con2:
            e2 = nengo.Ensemble(nengo.LIF(1), 1)
            assert e2 in con2.objs.values()
            assert not e2 in con1.objs.values()

            with con3:
                e3 = nengo.Ensemble(nengo.LIF(1), 1)
                assert e3 in con3.objs.values()
                assert not e3 in con2.objs.values()
                assert not e3 in con1.objs.values()

            e4 = nengo.Ensemble(nengo.LIF(1), 1)
            assert e4 in con2.objs.values()
            assert not e4 in con3.objs.values()

        e5 = nengo.Ensemble(nengo.LIF(1), 1)
        assert e5 in con1.objs.values()

    e6 = nengo.Ensemble(nengo.LIF(1), 1)
    assert not e6 in con1.objs.values()


def test_networks(Simulator):
    # TODO
    pass


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
