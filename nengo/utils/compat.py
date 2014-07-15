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
