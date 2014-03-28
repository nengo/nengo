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


def execfile(path, globals, locals=None):
    """Execute a Python script in the (mandatory) globals namespace.

    This is similar to the Python 2 builtin execfile, but it
    also works on Python 3, and ``globals`` is mandatory.
    This is because getting the calling frame's globals would
    be non-trivial, and it makes sense to be explicit about
    the namespace being modified.

    If ``locals`` is not specified, it will have the same value
    as ``globals``, as in the execfile builtin.
    """
    if locals is None:
        locals = globals
    with open(path, 'rb') as fp:
        source = fp.read()
    code = compile(source, path, "exec")
    exec(code, globals, locals)
