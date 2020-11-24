import pytest

import nengo
from nengo.base import NengoObjectParam
from nengo.exceptions import ValidationError


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
        n_in = NengoObjectParam("n_in", nonzero_size_in=True)
        n_out = NengoObjectParam("n_out", nonzero_size_out=True)

    inst = Test()
    with nengo.Network():
        n_in = nengo.Node(output=lambda t: t)
        n_out = nengo.Node(output=lambda t, x: None, size_in=1)
        probe = nengo.Probe(n_in)

        with pytest.raises(ValidationError):
            inst.n_in = n_in
        with pytest.raises(ValidationError):
            inst.n_out = n_out
        with pytest.raises(ValidationError):
            inst.n_out = probe

        inst.n_in = n_out
        inst.n_out = n_in
