import pickle
import tempfile

import pytest

import nengo
from nengo.base import NengoObjectParam
from nengo.exceptions import ValidationError


def test_nengoobjectparam():
    """NengoObjectParam must be a Nengo object and is readonly by default."""
    class Test(object):
        nop = NengoObjectParam('nop')
    inst = Test()

    # Must be a Nengo object
    with pytest.raises(ValidationError):
        inst.nop = 'a'

    # Can set it once
    a = nengo.Ensemble(10, dimensions=2, add_to_container=False)
    inst.nop = a.neurons
    assert inst.nop is a.neurons

    # Can't set it twice
    with pytest.raises(ValidationError):
        inst.nop = a


def test_nengoobjectparam_nonzero():
    """Can check that objects have nonzero size in/out."""
    class Test(object):
        nin = NengoObjectParam('nin', nonzero_size_in=True)
        nout = NengoObjectParam('nout', nonzero_size_out=True)

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


def test_pickle():
    with nengo.Network():
        a = nengo.Ensemble(10, 3)

    with tempfile.TemporaryFile() as f:
        with pytest.raises(NotImplementedError):
            pickle.dump(a, f)
        with pytest.raises(NotImplementedError):
            pickle.dump(a[:2], f)
