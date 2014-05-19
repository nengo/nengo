from __future__ import absolute_import

import collections
import sys

# Only test for Python 2 so that we have less changes for Python 4
PY2 = sys.version_info[0] == 2

# OrderedDict was introduced in Python 2.7 so for 2.6 we use the PyPI package
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
assert OrderedDict

# If something's changed from Python 2 to 3, we handle that here
if PY2:
    from StringIO import StringIO
    string_types = (str, unicode)
    int_types = (int, long)
    range = xrange

    # No iterkeys; use ``for key in dict:`` instead
    iteritems = lambda d: d.iteritems()
    itervalues = lambda d: d.itervalues()

    # We have to put this in an exec call because it's a syntax error in Py3+
    exec('def reraise(tp, value, tb):\n raise tp, value, tb')

else:
    from io import StringIO
    string_types = (str,)
    int_types = (int,)
    range = range

    # No iterkeys; use ``for key in dict:`` instead
    iteritems = lambda d: iter(d.items())
    itervalues = lambda d: iter(d.values())

    def reraise(tp, value, tb):
        raise value.with_traceback(tb)

assert StringIO


def is_integer(obj):
    return isinstance(obj, int_types)


def is_iterable(obj):
    return isinstance(obj, collections.Iterable)


def is_number(obj, check_complex=False):
    types = (float, complex) if check_complex else (float,)
    return isinstance(obj, types + int_types)


def is_string(obj):
    return isinstance(obj, string_types)


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


def with_metaclass(meta, *bases):
    """Function for creating a class with a metaclass.

    The syntax for this changed between Python 2 and 3.
    Code snippet from Armin Ronacher:
    http://lucumr.pocoo.org/2013/5/21/porting-to-python-3-redux/
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__

        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})


def groupby(objects, key, hashable=None, force_list=True):
    """Group objects based on a key.

    Unlike `itertools.groupby`, this function does not require the input
    to be sorted.

    Parameters
    ----------
    objects : Iterable
        The objects to be grouped.
    key : callable
        The key function by which to group the objects. If
        `key(obj1) == key(obj2)` then `obj1` and `obj2` are in the same group,
        otherwise they are not.
    hashable : boolean (optional)
        Whether to use the key's hash to determine equality. By default, this
        will be determined by calling `key` on the first item in `objects`, and
        if it is hashable, the hash will be used. Using a hash is faster, but
        not possible for all keys.
    force_list : boolean (optional)
        Whether to force the returned `key_groups` iterator, as well as the
        `group` iterator in each `(key, group)` pair, to be lists.

    Returns
    -------
    keygroups : iterable
        An iterable of `(key, group)` pairs, where `key` is the key used for
        grouping, and `group` is an iterable of the items in the group. The
        nature of the iterables depends on the value of `force_list`.
    """
    import itertools

    if hashable is None:
        # get first item without advancing iterator, and see if key is hashable
        objects, objects2 = itertools.tee(iter(objects))
        item0 = next(objects2)
        hashable = isinstance(key(item0), collections.Hashable)

    if hashable:
        # use a dictionary to sort by hash (faster)
        groups = {}
        for obj in objects:
            groups.setdefault(key(obj), []).append(obj)
        return list(groups.items()) if force_list else iteritems(groups)
    else:
        keygroupers = itertools.groupby(sorted(objects, key=key), key=key)
        if force_list:
            return [(k, [v for v in g]) for k, g in keygroupers]
        else:
            return keygroupers
