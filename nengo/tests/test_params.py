import numpy as np
import pytest

from nengo import params
from nengo.exceptions import ObsoleteError, ValidationError
from nengo.utils.compat import PY2


def test_default():
    """A default value is immediately available, but can be overridden."""

    class Test(object):
        p = params.Parameter('p', default=1)

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
        m = params.Parameter('m', default=1, optional=False)
        o = params.Parameter('o', default=1, optional=True)

    inst = Test()
    with pytest.raises(ValidationError):
        inst.m = None
    assert inst.m == 1
    inst.o = None
    assert inst.o is None


def test_readonly():
    """Readonly Parameters can only be set once."""

    class Test(object):
        p = params.Parameter('p', default=1, readonly=False)
        r = params.Parameter('r', default=None, readonly=True)

    inst = Test()
    assert inst.p == 1
    assert inst.r is None
    inst.p = 2
    inst.r = 'set'
    assert inst.p == 2
    assert inst.r == 'set'
    inst.p = 3
    with pytest.raises(ValidationError):
        inst.r = 'set again'
    assert inst.p == 3
    assert inst.r == 'set'


def test_obsoleteparam():
    """ObsoleteParams must not be set."""

    class Test(object):
        ab = params.ObsoleteParam('ab', "msg")

    inst = Test()

    # cannot be read
    with pytest.raises(ObsoleteError):
        inst.ab

    # can only be assigned Unconfigurable
    inst.ab = params.Unconfigurable
    with pytest.raises(ObsoleteError):
        inst.ab = True


def test_boolparam():
    """BoolParams can only be booleans."""

    class Test(object):
        bp = params.BoolParam('bp', default=False)

    inst = Test()
    assert not inst.bp
    inst.bp = True
    assert inst.bp
    with pytest.raises(ValidationError):
        inst.bp = 1


def test_numberparam():
    """NumberParams can be numbers constrained to a range."""

    class Test(object):
        np = params.NumberParam('np', default=1.0)
        np_l = params.NumberParam('np_l', default=1.0, low=0.0)
        np_h = params.NumberParam('np_h', default=-1.0, high=0.0)
        np_lh = params.NumberParam('np_lh', default=1.0, low=-1.0, high=1.0)

    inst = Test()

    # defaults
    assert inst.np == 1.0
    assert inst.np_l == 1.0
    assert inst.np_h == -1.0
    assert inst.np_lh == 1.0

    # respect low boundaries
    inst.np = -10
    with pytest.raises(ValidationError):
        inst.np_l = -10
    with pytest.raises(ValidationError):
        inst.np_lh = -10
    assert inst.np == -10
    assert inst.np_l == 1.0
    assert inst.np_lh == 1.0
    # equal to the low boundary is ok though!
    inst.np_lh = -1.0
    assert inst.np_lh == -1.0

    # respect high boundaries
    inst.np = 10
    with pytest.raises(ValidationError):
        inst.np_h = 10
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        inst.np = 'a'


def test_intparam():
    """IntParams are like NumberParams but must be an int."""
    class Test(object):
        ip = params.IntParam('ip', default=1, low=0, high=2)

    inst = Test()
    assert inst.ip == 1
    with pytest.raises(ValidationError):
        inst.ip = -1
    with pytest.raises(ValidationError):
        inst.ip = 3
    with pytest.raises(ValidationError):
        inst.ip = 'a'


def test_stringparam():
    """StringParams must be strings (bytes or unicode)."""
    class Test(object):
        sp = params.StringParam('sp', default="Hi")

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
    with pytest.raises(ValidationError):
        inst.sp = 1


def test_enumparam():
    class Test(object):
        ep = params.EnumParam('ep', default='a', values=('a', 'b', 'c'))

    inst = Test()
    assert inst.ep == 'a'
    inst.ep = 'b'
    assert inst.ep == 'b'
    inst.ep = 'c'
    assert inst.ep == 'c'
    inst.ep = 'A'
    assert inst.ep == 'a'
    with pytest.raises(ValueError):
        inst.ep = 'd'
    with pytest.raises(ValueError):
        inst.ep = 3


def test_dictparam():
    """DictParams must be dictionaries."""
    class Test(object):
        dp = params.DictParam('dp', default={'a': 1})

    inst1 = Test()
    assert inst1.dp == {'a': 1}
    inst1.dp['b'] = 2

    # The default dict is mutable -- other instances will get the same dict
    inst2 = Test()
    assert inst2.dp == {'a': 1, 'b': 2}

    # Non-dicts no good
    with pytest.raises(ValidationError):
        inst2.dp = [('a', 1), ('b', 2)]


def test_ndarrayparam():
    """NdarrayParams must be able to be made into float ndarrays."""
    class Test(object):
        ndp = params.NdarrayParam('ndp', default=None, shape=('*',))
        ella = params.NdarrayParam('ella', default=None, shape=(3, '...'))
        ellb = params.NdarrayParam('ellb', default=None, shape=(3, '...', 2))

    inst = Test()
    inst.ndp = np.ones(10)
    assert np.all(inst.ndp == np.ones(10))
    # Dimensionality too low
    with pytest.raises(ValidationError):
        inst.ndp = 0
    # Dimensionality too high
    with pytest.raises(ValidationError):
        inst.ndp = np.ones((1, 1))
    # Must be convertible to float array
    with pytest.raises(ValidationError):
        inst.ndp = 'a'

    inst.ella = np.ones((3,))
    inst.ella = np.ones((3, 1))
    inst.ella = np.ones((3, 2, 3))
    with pytest.raises(ValueError):
        inst.ella = np.ones(4)
    inst.ellb = np.ones((3, 2))
    inst.ellb = np.ones((3, 1, 2))
    with pytest.raises(ValueError):
        inst.ellb = np.ones(3)


def test_ndarrayparam_sample_shape():
    """sample_shape dictates the shape of the sample that can be set."""
    class Test(object):
        ndp = params.NdarrayParam('ndp', default=None, shape=[10, 'd2'])
        d2 = 3

    inst = Test()
    # Must be shape (4, 10)
    inst.ndp = np.ones((10, 3))
    assert np.all(inst.ndp == np.ones((10, 3)))
    with pytest.raises(ValidationError):
        inst.ndp = np.ones((3, 10))
    assert np.all(inst.ndp == np.ones((10, 3)))


def test_functionparam():
    """FunctionParam must be a function, and accept one scalar argument."""
    class Test(object):
        fp = params.FunctionParam('fp', default=None)

    inst = Test()
    assert inst.fp is None
    inst.fp = np.sin
    assert inst.fp.function is np.sin
    assert inst.fp.size == 1
    # Not OK: requires two args
    with pytest.raises(ValidationError):
        inst.fp = lambda x, y: x + y
    # Not OK: not a function
    with pytest.raises(ValidationError):
        inst.fp = 0


def test_deferrable():
    """The value Deferrable of  is either set to a fixed value or
    determined at some later time via a function call."""

    mock = params.Deferral(lambda x: 2. * x)

    class Test(object):
        p_def_fn = params.Deferrable(
            params.NumberParam('p_def_fn', readonly=False, low=0., high=10.),
            default=mock)
        p_def_num = params.Deferrable(params.NumberParam(
            'p_def_num', default=2., low=0., high=10.))

    inst = Test()

    # defaults
    assert inst.p_def_fn is mock
    assert inst.p_def_num == 2.

    # assign numbers
    inst.p_def_fn = 3.
    inst.p_def_num = 4.
    assert inst.p_def_fn == 3.
    assert inst.p_def_num == 4.

    # assign deferral
    inst.p_def_fn = mock
    inst.p_def_num = mock
    assert inst.p_def_fn is mock
    assert inst.p_def_num is mock

    # assign invalid values
    with pytest.raises(ValueError):
        inst.p_def_fn = -1.
    with pytest.raises(ValueError):
        inst.p_def_num = 11.
    with pytest.raises(ValueError):
        inst.p_def_fn = lambda x: 2. * x


def test_undeferred():
    class DeferralFn(params.Deferral):
        def __init__(self):
            super(DeferralFn, self).__init__()
            self.n_calls = 0

        def default_fn(self, *args, **kwargs):
            self.n_calls += 1
            return 42.

    mock_sim = type('MockSim', (), {})
    DeferralFn.register(mock_sim, lambda: 1)

    deferral = DeferralFn()

    class Test(object):
        p = params.Deferrable(params.NumberParam('p'), default=deferral)

    inst = Test()

    undeferred = params.Undeferred(inst, None)
    assert undeferred.p == 42.
    assert undeferred.p == 42.  # check twice to check caching
    assert deferral.n_calls == 1

    assert hash(inst) == hash(undeferred)
    assert inst == undeferred

    undeferred_mocksim = params.Undeferred(inst, mock_sim)
    assert undeferred_mocksim.p == 1

    inst.p = params.Deferral(lambda: 'str')
    with pytest.raises(ValueError):
        params.Undeferred(inst, None).p


def test_deferral_classes_have_independent_sim_specific_dicts():
    class DeferralA(params.Deferral):
        pass

    class DeferralB(params.Deferral):
        pass

    mock_sim = type('MockSim', (), {})
    fn_a = lambda: 'a'
    fn_b = lambda: 'b'

    DeferralA.register(mock_sim, fn_a)
    DeferralB.register(mock_sim, fn_b)

    assert DeferralA().get_deferral_fn(mock_sim) == fn_a
    assert DeferralB().get_deferral_fn(mock_sim) == fn_b
