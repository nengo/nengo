"""Utility functions related to the inspect module."""

from __future__ import absolute_import

from collections import namedtuple
import inspect

CheckedCall = namedtuple('CheckedCall', ('value', 'invoked'))


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
