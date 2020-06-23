import sys
import weakref

import numpy as np
import pytest

from nengo.utils.stdlib import (
    checked_call,
    groupby,
    Timer,
    WeakKeyDefaultDict,
    WeakKeyIDDictionary,
    WeakSet,
)


def test_checked_call():
    def func1(a):
        return a

    def func2(a, b=0, **kwargs):
        return a + b

    def func3(a, b=0, c=0, *args, **kwargs):
        return a + b + c + sum(args)

    func4 = lambda x=[0]: sum(x)

    class A:
        def __call__(self, a, b):
            return a + b

    assert checked_call(func1) == (None, False)
    assert checked_call(func1, 1) == (1, True)
    assert checked_call(func1, 1, 2) == (None, False)
    assert checked_call(func1, 1, two=2) == (None, False)

    assert checked_call(func2) == (None, False)
    assert checked_call(func2, 1) == (1, True)
    assert checked_call(func2, 1, 2) == (3, True)
    assert checked_call(func2, 1, 2, three=3) == (3, True)
    assert checked_call(func2, 1, 2, 3) == (None, False)

    assert checked_call(func3) == (None, False)
    assert checked_call(func3, 1) == (1, True)
    assert checked_call(func3, 1, 2) == (3, True)
    assert checked_call(func3, 1, 2, 3) == (6, True)
    assert checked_call(func3, 1, 2, 3, 4, 5, 6) == (21, True)

    assert checked_call(func4) == (0, True)
    assert checked_call(func4, [1, 2]) == (3, True)
    assert checked_call(func4, [1], 2) == (None, False)

    assert checked_call(A(), 1) == (None, False)
    assert checked_call(A(), 1, 2) == (3, True)
    assert checked_call(A(), 1, 2, 3) == (None, False)

    assert checked_call(np.sin) == (None, False)
    assert checked_call(np.sin, 0) == (0, True)
    assert checked_call(np.sin, 0, np.array([1.0])) == (np.array([0.0]), True)
    assert checked_call(np.sin, 0, np.array([1.0]), 1) == (None, False)


def test_checked_call_errors():
    class A:
        def __call__(self, a):
            raise NotImplementedError()

    assert checked_call(A()) == (None, False)
    with pytest.raises(NotImplementedError):
        checked_call(A(), 1)

    assert checked_call(np.sin, 1, 2, 3) == (None, False)
    with pytest.raises(ValueError):
        checked_call(lambda x: np.sin(1, 2, 3), 1)


@pytest.mark.parametrize(
    "hashable, force_list", [(False, False), (False, True), (True, False), (True, True)]
)
def test_groupby(hashable, force_list, rng):
    if hashable:
        keys = list(range(1, 5))
    else:
        keys = [[0, 0], [0, 1], [1, 0], [1, 1]]

    keys = sorted(keys)

    # make groups and pairs
    groups = [rng.randn(rng.randint(5, 10)) for _ in keys]

    pairs = []
    for key, group in zip(keys, groups):
        pairs.extend((key, value) for value in group)

    # shuffle pairs
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]

    # call groupby
    keygroups = groupby(pairs, lambda p: p[0], force_list=force_list)

    keys2 = sorted([x[0] for x in keygroups])
    assert keys2 == keys

    for key2, keygroup2 in keygroups:
        group = groups[keys.index(key2)]
        group2 = [x[1] for x in keygroup2]
        assert sorted(group2) == sorted(group)


def test_timer():
    with Timer() as timer:
        for i in range(1000):
            2 + 2
    assert timer.duration > 0.0
    assert timer.duration < 1.0  # Pretty bad worst case


class C:
    def method(self):
        pass


def test_weakkeydefaultdict():
    factory = lambda: "default"
    d = WeakKeyDefaultDict(factory)
    o = C()

    assert len(d) == 0
    assert d[o] == "default"

    d[o] = "changed"
    assert len(d) == 1
    assert d[o] == "changed"

    del d[o]
    assert len(d) == 0
    assert o not in d

    d[o] = "changed"
    for index in d:
        assert index == o
    del index
    del o
    assert len(d) == 0


def test_weakkeyiddictionary_init():
    obj = C()
    val = 364

    # construct from dictionary
    d = WeakKeyIDDictionary({obj: val})
    assert d[obj] == val

    # construct from another WeakKeyIDDictionary
    d2 = WeakKeyIDDictionary(d)
    assert d2[obj] == val


def test_weakkeydict_popitem(key1=C(), key2=C(), val1="v1", val2="v2"):
    d = WeakKeyIDDictionary()
    d[key1] = val1
    d[key2] = val2

    assert len(d) == 2
    k1, v1 = d.popitem()
    assert len(d) == 1
    assert k1 in (key1, key2) and v1 is (val1 if k1 is key1 else val2)

    k2, v2 = d.popitem()
    assert len(d) == 0
    assert k2 is (key2 if k1 is key1 else key1) and v2 is (val1 if k2 is key1 else val2)


def test_weakkeydict_setdefault(key=C(), value1="v1", value2="v2"):
    d = WeakKeyIDDictionary()
    o = d.setdefault(key, value1)
    assert o is value1
    assert key in d
    assert d.get(key) is value1
    assert d[key] is value1

    o = d.setdefault(key, value2)
    assert o is value1
    assert key in d
    assert d.get(key) is value1
    assert d[key] is value1


def test_weakkeydict_update():
    """This exercises d.update(), len(d), d.keys(), in d,  d.get(), d[]."""
    in_d = {C(): 1, C(): 2, C(): 3}
    d = WeakKeyIDDictionary()
    d.update(in_d)

    assert len(d) == len(in_d)
    for k in d.keys():
        assert k in in_d, "mysterious new key appeared in weak dict"
        v = in_d.get(k)
        assert v is d[k] and v is d.get(k)
    for k in in_d.keys():
        assert k in d, "original key disappeared in weak dict"
        v = in_d[k]
        assert v is d[k] and v is d.get(k)


def test_weakkeyiddict_contains_none():
    assert (None in WeakKeyIDDictionary()) is False


def test_weakkeyiddict_iter_functions():
    """Tests iterkeys and iteritems"""
    in_d = {C(): 1, C(): 2, C(): 3}
    d = WeakKeyIDDictionary(in_d)

    keys = list(in_d)
    for key in d.iterkeys():
        assert key in keys, "WeakKeyIDDictionary has extra key %d" % (key,)
        keys.remove(key)

    assert len(keys) == 0, "Keys not found in WeakKeyIDDictionary: %r" % (keys,)

    # copy, otherwise `d` will change size during iteration as references are removed
    in_d2 = dict(in_d)
    for k, v in d.iteritems():
        assert k in in_d
        assert in_d2.pop(k) == v

    assert len(in_d2) == 0, "Keys not found in WeakKeyIDDictionary: %r" % (in_d2,)


def test_weakkeydict_delitem():
    d = WeakKeyIDDictionary()
    o1 = C()
    o2 = C()
    d[o1] = "something"
    d[o2] = "something"
    assert len(d) == 2
    del d[o1]
    assert len(d) == 1
    assert list(d.keys()) == [o2]


def test_weakkeydict_bad_delitem():
    d = WeakKeyIDDictionary()
    o = C()

    # An attempt to delete an object that isn't there should raise KeyError.
    with pytest.raises(KeyError):
        del d[o]
    with pytest.raises(KeyError):
        d[o]

    # If a key isn't of a weakly referencable type, __getitem__ and
    # __setitem__ raise TypeError.  __delitem__ should too.
    with pytest.raises(TypeError):
        del d[13]
    with pytest.raises(TypeError):
        d[13]
    with pytest.raises(TypeError):
        d[13] = 13


def test_weakkeydict_frees_values():
    d = WeakKeyIDDictionary()
    k = C()
    v = C()
    d[k] = v
    weak_v = weakref.ref(v)
    del v
    assert sys.getrefcount(weak_v()) > 1  # function argument might make it > 1
    del k
    v = weak_v()
    assert v is None, "Value in WeakKeyIDDictionary not garbage collected."


def test_weakset():
    s = WeakSet()
    k = C()

    s.add(k)
    assert len(s) == 1
    assert k in s

    s.discard(k)
    assert len(s) == 0
    assert k not in s

    s.add(k)
    del k
    assert len(s) == 0


def test_weakset_init():
    k = C()

    # init from a list
    s = WeakSet([k])
    assert k in s

    # init from another weakset
    s2 = WeakSet(s)
    assert k in s2
