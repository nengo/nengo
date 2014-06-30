from collections import Counter
import logging

import numpy as np
import pytest

import nengo
from nengo import params
from nengo.utils.compat import PY2
from nengo.utils.distributions import UniformHypersphere
from nengo.neurons import LIF
from nengo.synapses import Lowpass

logger = logging.getLogger(__name__)


def test_default():
    """A default value is immediately available, but can be overridden."""

    class Test(object):
        p = params.Parameter(default=1)

    inst1 = Test()
    inst2 = Test()
    assert inst1.p == 1
    assert inst2.p == 1
    inst1.p = 'a'
    assert inst1.p == 'a'
    assert inst2.p == 1


def test_optional():
    """Optional Parameters can bet set to None."""

    class Test(object):
        m = params.Parameter(default=1, optional=False)
        o = params.Parameter(default=1, optional=True)

    inst = Test()
    with pytest.raises(ValueError):
        inst.m = None
    assert inst.m == 1
    inst.o = None
    assert inst.o is None


def test_readonly():
    """Readonly Parameters can only be set once."""

    class Test(object):
        p = params.Parameter(default=1, readonly=False)
        r = params.Parameter(default=None, readonly=True)

    inst = Test()
    assert inst.p == 1
    assert inst.r is None
    inst.p = 2
    inst.r = 'set'
    assert inst.p == 2
    assert inst.r == 'set'
    inst.p = 3
    with pytest.raises(ValueError):
        inst.r = 'set again'
    assert inst.p == 3
    assert inst.r == 'set'


def test_readonly_assert():
    """Readonly Parameters must default to None."""

    with pytest.raises(AssertionError):
        class Test(object):
            p = params.Parameter(default=1, readonly=True)


def test_boolparam():
    """BoolParams can only be booleans."""

    class Test(object):
        bp = params.BoolParam(default=False)

    inst = Test()
    assert not inst.bp
    inst.bp = True
    assert inst.bp
    with pytest.raises(ValueError):
        inst.bp = 1


def test_numberparam():
    """NumberParams can be numbers constrained to a range."""

    class Test(object):
        np = params.NumberParam(default=1.0)
        np_l = params.NumberParam(default=1.0, low=0.0)
        np_h = params.NumberParam(default=-1.0, high=0.0)
        np_lh = params.NumberParam(default=1.0, low=-1.0, high=1.0)

    inst = Test()

    # defaults
    assert inst.np == 1.0
    assert inst.np_l == 1.0
    assert inst.np_h == -1.0
    assert inst.np_lh == 1.0

    # respect low boundaries
    inst.np = -10
    with pytest.raises(ValueError):
        inst.np_l = -10
    with pytest.raises(ValueError):
        inst.np_lh = -10
    assert inst.np == -10
    assert inst.np_l == 1.0
    assert inst.np_lh == 1.0
    # equal to the low boundary is ok though!
    inst.np_lh = -1.0
    assert inst.np_lh == -1.0

    # respect high boundaries
    inst.np = 10
    with pytest.raises(ValueError):
        inst.np_h = 10
    with pytest.raises(ValueError):
        inst.np_lh = 10
    assert inst.np == 10
    assert inst.np_h == -1.0
    assert inst.np_lh == -1.0
    # equal to the high boundary is ok though!
    inst.np_lh = 1.0
    assert inst.np_lh == 1.0

    # must be a number!
    with pytest.raises(ValueError):
        inst.np = 'a'


def test_numberparam_assert():
    """Malformed NumberParams."""

    # low > high = bad
    with pytest.raises(AssertionError):
        class TestLH(object):
            np = params.NumberParam(default=0, low=1, high=-1)

    # default must be in range
    with pytest.raises(AssertionError):
        class TestDH(object):
            np = params.NumberParam(default=1.0, high=0.0)
    with pytest.raises(AssertionError):
        class TestDL(object):
            np = params.NumberParam(default=-1.0, low=0.0)


def test_intparam():
    """IntParams are like NumberParams but must be an int."""
    class Test(object):
        ip = params.IntParam(default=1, low=0, high=2)

    inst = Test()
    assert inst.ip == 1
    with pytest.raises(ValueError):
        inst.ip = -1
    with pytest.raises(ValueError):
        inst.ip = 3
    with pytest.raises(ValueError):
        inst.ip = 'a'


def test_stringparam():
    """StringParams must be strings (bytes or unicode)."""
    class Test(object):
        sp = params.StringParam(default="Hi")

    inst = Test()
    assert inst.sp == "Hi"

    # Bytes OK on Python 2
    if PY2:
        inst.sp = b"hello"
        assert inst.sp == b"hello"
    # Unicode OK on both
    inst.sp = u"goodbye"
    assert inst.sp == u"goodbye"

    # Non-strings no good
    with pytest.raises(ValueError):
        inst.sp = 1


def test_listparam():
    """ListParams must be lists."""
    class Test(object):
        lp = params.ListParam(default=[1])

    inst1 = Test()
    assert inst1.lp == [1]
    inst1.lp.append(2)

    # The default list is mutable -- other instances will get the same list
    inst2 = Test()
    assert len(inst2.lp) == 2

    # Non-lists no good
    with pytest.raises(ValueError):
        inst2.lp = (1, 2)


def test_distributionparam():
    """DistributionParams can be distributions or samples."""
    class Test(object):
        dp = params.DistributionParam(default=None, sample_shape=['*', '*'])

    inst = Test()
    inst.dp = UniformHypersphere()
    assert isinstance(inst.dp, UniformHypersphere)
    inst.dp = np.array([[1], [2], [3]])
    assert np.all(inst.dp == np.array([[1], [2], [3]]))
    with pytest.raises(ValueError):
        inst.dp = 'a'


def test_distributionparam_sample_shape():
    """sample_shape dictates the shape of the sample that can be set."""
    class Test(object):
        dp = params.DistributionParam(default=None, sample_shape=['d1', 'd2'])
        d1 = 4
        d2 = 10

    inst = Test()
    # Distributions are still cool
    inst.dp = UniformHypersphere()
    assert isinstance(inst.dp, UniformHypersphere)
    # Must be shape (4, 10)
    inst.dp = np.ones((4, 10))
    assert np.all(inst.dp == np.ones((4, 10)))
    with pytest.raises(ValueError):
        inst.dp = np.ones((10, 4))
    assert np.all(inst.dp == np.ones((4, 10)))


def test_neurontypeparam():
    """NeuronTypeParam must be a neuron type."""
    class Test(object):
        ntp = params.NeuronTypeParam(default=None)

    inst = Test()
    inst.ntp = LIF()
    assert isinstance(inst.ntp, LIF)
    with pytest.raises(ValueError):
        inst.ntp = 'a'


def test_neurontypeparam_probeable():
    """NeuronTypeParam can update a probeable list."""
    class Test(object):
        ntp = params.NeuronTypeParam(default=None, optional=True)
        probeable = ['output']

    inst = Test()
    assert inst.probeable == ['output']
    inst.ntp = LIF()
    assert Counter(inst.probeable) == Counter(inst.ntp.probeable + ['output'])
    # The first element is important,  as it's the default
    assert inst.probeable[0] == 'output'
    # Setting it again should result in the same list
    inst.ntp = LIF()
    assert Counter(inst.probeable) == Counter(inst.ntp.probeable + ['output'])
    assert inst.probeable[0] == 'output'
    # Unsetting it should clear the list appropriately
    inst.ntp = None
    assert inst.probeable == ['output']


def test_synapseparam():
    """SynapseParam must be a Synapse, and converts numbers to LowPass."""
    class Test(object):
        sp = params.SynapseParam(default=Lowpass(0.1))

    inst = Test()
    assert isinstance(inst.sp, Lowpass)
    assert inst.sp.tau == 0.1
    # Number are converted to LowPass
    inst.sp = 0.05
    assert isinstance(inst.sp, Lowpass)
    assert inst.sp.tau == 0.05
    # None has meaning
    inst.sp = None
    assert inst.sp is None
    # Non-synapse not OK
    with pytest.raises(ValueError):
        inst.sp = 'a'


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
