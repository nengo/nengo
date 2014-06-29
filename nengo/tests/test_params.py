import logging

import pytest

import nengo
from nengo import params

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


def test_numberparam_assert():
    """Malformed NumberParams."""

    # low > high = bad
    with pytest.raises(AssertionError):
        class Test(object):
            np = params.NumberParam(default=0, low=1, high=-1)

    # default must be in range
    with pytest.raises(AssertionError):
        class Test(object):
            np = params.NumberParam(default=1.0, high=0.0)
    with pytest.raises(AssertionError):
        class Test(object):
            np = params.NumberParam(default=-1.0, low=0.0)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
