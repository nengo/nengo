import numpy as np
import pytest

from nengo import params
from nengo.utils.compat import PY2


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


def test_obsoleteparam():
    """ObsoleteParams must not be set."""

    class Test(object):
        ab = params.ObsoleteParam(123, "msg")

    inst = Test()

    # cannot be read
    with pytest.raises(ValueError):
        inst.ab

    # can only be assigned Unconfigurable
    inst.ab = params.Unconfigurable
    with pytest.raises(ValueError):
        inst.ab = True


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

    # ensure scalar array works
    inst.np = np.array(2.0)
    assert inst.np == 2.0

    # must be a number!
    with pytest.raises(ValueError):
        inst.np = 'a'


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


def test_dictparam():
    """DictParams must be dictionaries."""
    class Test(object):
        dp = params.DictParam(default={'a': 1})

    inst1 = Test()
    assert inst1.dp == {'a': 1}
    inst1.dp['b'] = 2

    # The default dict is mutable -- other instances will get the same dict
    inst2 = Test()
    assert inst2.dp == {'a': 1, 'b': 2}

    # Non-dicts no good
    with pytest.raises(ValueError):
        inst2.dp = [('a', 1), ('b', 2)]


def test_ndarrayparam():
    """NdarrayParams must be able to be made into float ndarrays."""
    class Test(object):
        ndp = params.NdarrayParam(default=None, shape=('*',))

    inst = Test()
    inst.ndp = np.ones(10)
    assert np.all(inst.ndp == np.ones(10))
    # Dimensionality too low
    with pytest.raises(ValueError):
        inst.ndp = 0
    # Dimensionality too high
    with pytest.raises(ValueError):
        inst.ndp = np.ones((1, 1))
    # Must be convertible to float array
    with pytest.raises(ValueError):
        inst.ndp = 'a'


def test_ndarrayparam_sample_shape():
    """sample_shape dictates the shape of the sample that can be set."""
    class Test(object):
        ndp = params.NdarrayParam(default=None, shape=[10, 'd2'])
        d2 = 3

    inst = Test()
    # Must be shape (4, 10)
    inst.ndp = np.ones((10, 3))
    assert np.all(inst.ndp == np.ones((10, 3)))
    with pytest.raises(ValueError):
        inst.ndp = np.ones((3, 10))
    assert np.all(inst.ndp == np.ones((10, 3)))


def test_functionparam():
    """FunctionParam must be a function, and accept one scalar argument."""
    class Test(object):
        fp = params.FunctionParam(default=None)

    inst = Test()
    assert inst.fp is None
    inst.fp = np.sin
    assert inst.fp.function is np.sin
    assert inst.fp.size == 1
    # Not OK: requires two args
    with pytest.raises(TypeError):
        inst.fp = lambda x, y: x + y
    # Not OK: not a function
    with pytest.raises(ValueError):
        inst.fp = 0
