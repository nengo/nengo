import inspect

import numpy as np


def in_stack(function):
    """Check whether the given function is in the call stack"""
    codes = [record[0].f_code for record in inspect.stack()]
    return function.__code__ in codes


def register(callback):
    """A decorator that registers the wrapped function with callback(func)."""
    def wrapper(func):
        callback(func)
        return func
    return wrapper


def random_maxint(rng):
    """Returns rng.randint(x) where x is the maximum 32-bit integer."""
    return rng.randint(np.iinfo(np.int32).max)
