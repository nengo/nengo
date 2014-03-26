from __future__ import absolute_import
import collections
try:
    from collections import OrderedDict  # noqa: F401
except ImportError:
    from ordereddict import OrderedDict  # noqa: F401
import sys

IS_PYTHON3 = sys.version_info[0] == 3


def is_callable(obj):
    return isinstance(obj, collections.Callable)


def is_integer(obj):
    return isinstance(obj, int if IS_PYTHON3 else (int, long))


def is_iterable(obj):
    return isinstance(obj, collections.Iterable)


def is_number(obj, check_complex=False):
    types = (float, complex) if check_complex else float
    return isinstance(obj, types) or is_integer(obj)


def is_string(obj):
    types = str if IS_PYTHON3 else basestring
    return isinstance(obj, types)
