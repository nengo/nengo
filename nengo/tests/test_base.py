import pytest

import nengo
from nengo.base import FrozenObject, NengoObjectParam
from nengo.exceptions import ValidationError
from nengo.params import NumberParam


def test_nengoobjectparam():
    """NengoObjectParam must be a Nengo object and is readonly by default."""

    class Test:
        nop = NengoObjectParam("nop")

    inst = Test()

    # Must be a Nengo object
    with pytest.raises(ValidationError):
        inst.nop = "a"

    # Can set it once
    a = nengo.Ensemble(10, dimensions=2, add_to_container=False)
    inst.nop = a.neurons
    assert inst.nop == a.neurons

    # Can't set it twice
    with pytest.raises(ValidationError):
        inst.nop = a


def test_nengoobjectparam_nonzero():
    """Can check that objects have nonzero size in/out."""

    class Test:
        nin = NengoObjectParam("nin", nonzero_size_in=True)
        nout = NengoObjectParam("nout", nonzero_size_out=True)

    inst = Test()
    with nengo.Network():
        nin = nengo.Node(output=lambda t: t)
        nout = nengo.Node(output=lambda t, x: None, size_in=1)
        probe = nengo.Probe(nin)

        with pytest.raises(ValidationError):
            inst.nin = nin
        with pytest.raises(ValidationError):
            inst.nout = nout
        with pytest.raises(ValidationError):
            inst.nout = probe

        inst.nin = nout
        inst.nout = nin


def test_frozenobject_reprs():
    class TestFO(FrozenObject):
        a = NumberParam("a", default=3, readonly=True)
        b = NumberParam("b")

        def __init__(self, a, b=4):
            super().__init__()
            self.a = a
            self.b = b

    # test that parameters are only shown in repr if their values are different
    # than the default, for both parameter defaults and argument defaults
    assert repr(TestFO(3)) == "TestFO()"
    assert repr(TestFO(2)) == "TestFO(a=2)"
    assert repr(TestFO(2, b=3)) == "TestFO(a=2, b=3)"


def test_frozenobject_missing_arg_repr():
    class TestFO(FrozenObject):
        a = NumberParam("a", default=3, readonly=True)

        def __init__(self, a, b=4):
            super().__init__()
            self.a = a

    fobj = TestFO(3)
    assert repr(fobj).startswith("<TestFO at")
    assert fobj._argreprs == "Cannot find 'b'"
