import pytest

import nengo
from nengo.base import NengoObjectParam


def test_nengoobjectparam():
    """NengoObjectParam must be a Nengo object and is readonly by default."""
    class Test(object):
        nop = NengoObjectParam()

    inst = Test()
    assert inst.nop is None
    # Must be a Nengo object
    with pytest.raises(ValueError):
        inst.nop = 'a'
    a = nengo.Ensemble(10, dimensions=2, add_to_container=False)
    inst.nop = a.neurons
    assert inst.nop is a.neurons
    # Can't set it twice
    with pytest.raises(ValueError):
        inst.nop = a


def test_nengoobjectparam_disallow():
    """Can disallow specific Nengo objects."""
    class Test(object):
        nop = NengoObjectParam(disallow=[nengo.Connection])

    inst = Test()
    with nengo.Network():
        a = nengo.Ensemble(10, 2)
        b = nengo.Ensemble(10, 2)
        with pytest.raises(ValueError):
            inst.nop = nengo.Connection(a, b)
        inst.nop = b
        assert inst.nop is b


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
