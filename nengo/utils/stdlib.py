"""
Functions that extend the Python Standard Library.
"""

from __future__ import absolute_import

import collections
import inspect
import itertools
import os
import shutil

from .compat import iteritems

CheckedCall = collections.namedtuple('CheckedCall', ('value', 'invoked'))


def checked_call(func, *args, **kwargs):
    """Calls func(*args, **kwargs) and checks that invocation was successful.

    The namedtuple ``(value=func(*args, **kwargs), invoked=True)`` is returned
    if the call is successful. If an exception occurs inside of ``func``, then
    that exception will be raised. Otherwise, if the exception occurs as a
    result of invocation, then ``(value=None, invoked=False)`` is returned.

    Assumes that func is callable.
    """
    try:
        return CheckedCall(func(*args, **kwargs), True)
    except:
        tb = inspect.trace()
        if not len(tb) or tb[-1][0] is not inspect.currentframe():
            raise  # exception occurred inside func
    return CheckedCall(None, False)


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


# terminal_size was introduced in Python 3.3
if hasattr(os, 'terminal_size'):
    terminal_size = os.terminal_size
else:
    terminal_size = collections.namedtuple(
        'terminal_size', ['columns', 'lines'])


# get_terminal_size was introduced in Python 3.3
if hasattr(shutil, 'get_terminal_size'):
    get_terminal_size = shutil.get_terminal_size
else:
    def get_terminal_size(fallback=(80, 24)):
        w, h = fallback
        try:
            w = int(os.environ['COLUMNS'])
        except:
            pass
        try:
            h = int(os.environ['LINES'])
        except:
            pass
        return terminal_size(w, h)
