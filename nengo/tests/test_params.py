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

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
