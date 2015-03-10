import pickle
import tempfile

import pytest

import nengo
from nengo.base import NengoObjectParam


def test_nengoobjectparam():
    """NengoObjectParam must be a Nengo object and is readonly by default."""
    class Test(object):
        nop = NengoObjectParam()
    inst = Test()

    # Must be a Nengo object
    with pytest.raises(ValueError):
        inst.nop = 'a'

    # Can set it once
    a = nengo.Ensemble(10, dimensions=2, add_to_container=False)
    inst.nop = a.neurons
    assert inst.nop is a.neurons

    # Can't set it twice
    with pytest.raises(ValueError):
        inst.nop = a


def test_nengoobjectparam_nonzero():
    """Can check that objects have nonzero size in/out."""
    class Test(object):
        nin = NengoObjectParam(nonzero_size_in=True)
        nout = NengoObjectParam(nonzero_size_out=True)

    inst = Test()
    with nengo.Network():
        nin = nengo.Node(output=lambda t: t)
        nout = nengo.Node(output=lambda t, x: None, size_in=1)
        probe = nengo.Probe(nin)

        with pytest.raises(ValueError):
            inst.nin = nin
        with pytest.raises(ValueError):
            inst.nout = nout
        with pytest.raises(ValueError):
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


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
