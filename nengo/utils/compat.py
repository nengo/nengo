import collections
import os
import subprocess
import sys
import warnings

import numpy as np

# Only test for Python 2 so that we have less changes for Python 4
PY2 = sys.version_info[0] == 2

# If something's changed from Python 2 to 3, we handle that here
if PY2:  # pragma: no cover
    warnings.warn(
        "Python 2 compatibility has been dropped from Nengo as of "
        "version 3.0.0; nengo.utils.compat will be removed in the "
        "next minor release."
    )
    from cgi import escape as cgi_escape
    import cPickle as pickle
    import ConfigParser as configparser
    import inspect
    from itertools import izip_longest as zip_longest
    from StringIO import StringIO

    escape = lambda s, quote=True: cgi_escape(s, quote=quote)
    string_types = (str, unicode)
    int_types = (int, long)
    range = xrange

    # No iterkeys; use ``for key in dict:`` instead
    iteritems = lambda d: d.iteritems()
    itervalues = lambda d: d.itervalues()

    # We have to put this in an exec call because it's a syntax error in Py3+
    exec("def reraise(tp, value, tb):\n raise tp, value, tb")

    def ensure_bytes(s):
        if isinstance(s, unicode):
            return s.encode("utf-8")
        assert isinstance(s, bytes)
        return s

    FullArgSpec = collections.namedtuple(
        "FullArgSpec",
        [
            "args",
            "varargs",
            "varkw",
            "defaults",
            "kwonlyargs",
            "kwonlydefaults",
            "annotations",
        ],
    )

    def getfullargspec(func):
        argspec = inspect.getargspec(func)  # pylint: disable=deprecated-method
        return FullArgSpec(
            args=argspec.args,
            varargs=argspec.varargs,
            varkw=argspec.keywords,
            defaults=argspec.defaults,
            kwonlyargs=[],
            kwonlydefaults={},
            annotations={},
        )

    if sys.platform.startswith("win"):

        def replace(src, dst):
            # The Windows implementation of replace calls out to the shell
            # to do 'move /Y src dst' due to an odd bug in 32-bit versions
            # of Python 2.7. See https://github.com/nengo/nengo/pull/1107
            # for the bizarre details.
            with open(os.devnull, "w") as devnull:
                subprocess.check_call(
                    ["move", "/Y", src, dst], shell=True, stdout=devnull, stderr=devnull
                )

    else:

        def replace(src, dst):
            try:
                os.rename(src, dst)
            except OSError:
                os.remove(dst)
                os.rename(src, dst)

    class TextIO(StringIO):
        def write(self, data):
            if not isinstance(data, unicode):
                data = unicode(data, getattr(self, "_encoding", "UTF-8"), "replace")
            StringIO.write(self, data)

    class ResourceWarning(DeprecationWarning):
        """A warning about resource usage.

        Note that we subclass from DeprecationWarning here solely because
        DeprecationWarnings are filtered out by default in Python 2.7,
        while in Python 3.2+ both DeprecationWarnings and ResourceWarnings
        are filtered out. Subclassing from DeprecationWarning gives
        the same (or at least very similar) behavior in Python 2 and 3
        without having to modify filters in the warnings module.
        """


else:  # pragma: no cover
    import pickle
    import configparser
    from html import escape
    from inspect import getfullargspec
    from io import StringIO
    from itertools import zip_longest
    from os import replace

    TextIO = StringIO
    string_types = (str,)
    int_types = (int,)
    range = range
    ResourceWarning = ResourceWarning

    # No iterkeys; use ``for key in dict:`` instead
    iteritems = lambda d: iter(d.items())
    itervalues = lambda d: iter(d.values())

    def reraise(tp, value, tb):
        raise value.with_traceback(tb)

    def ensure_bytes(s):
        if isinstance(s, str):
            s = s.encode("utf-8")
        assert isinstance(s, bytes)
        return s


def is_integer(obj):  # pragma: no cover
    return isinstance(obj, int_types + (np.integer,))


def is_iterable(obj):  # pragma: no cover
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0  # 0-d arrays give error if iterated over
    else:
        return isinstance(obj, collections.Iterable)


def is_number(obj, check_complex=False):  # pragma: no cover
    types = (float, complex, np.number) if check_complex else (float, np.floating)
    return is_integer(obj) or isinstance(obj, types)


def is_string(obj):  # pragma: no cover
    return isinstance(obj, string_types)


def is_array(obj):  # pragma: no cover
    # np.generic allows us to return true for scalars as well as true arrays
    return isinstance(obj, (np.ndarray, np.generic))


def is_array_like(obj):  # pragma: no cover
    # While it's possible that there are some iterables other than list/tuple
    # that can be made into arrays, it's very likely that those arrays
    # will have dtype=object, which is likely to cause unexpected issues.
    return is_array(obj) or is_number(obj) or isinstance(obj, (list, tuple))
