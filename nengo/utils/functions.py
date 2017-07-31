from __future__ import absolute_import

from collections import OrderedDict

import numpy as np

from nengo.exceptions import ValidationError
from nengo.utils.compat import is_number, iteritems


def piecewise(data):
    """Create a piecewise constant function from a dictionary.

    Given an input of data={0: 0, 0.5: 1, 0.75: -1, 1: 0} this will generate a
    function that returns 0 up until t=0.5, then outputs a 1 until t=0.75,
    then a -1 until t=1, and then returns 0 after that. This is meant as a
    shortcut for::

        def function(t):
            if t < 0.5:
                return 0
            elif t < 0.75
                return 1
            elif t < 1:
                return -1
            else:
                return 0

    The keys in the dictionary must be times (floats or ints). The values in
    the data dictionary can be floats, lists, or functions that return
    floats or lists. All lists must be of the same length.

    For times before the first specified time, it will default to zero (of
    the correct length). This means the above example can be simplified to::

        piecewise({0.5: 1, 0.75: -1, 1: 0})

    Parameters
    ----------
    data : dict
        The values to change to. Keys are the beginning time for the value.
        Values can be int, float, list, or functions that return those.

    Returns
    -------
    function:
        A function that takes a variable t and returns the corresponding
        value from the dictionary.

    Examples
    --------

      >>> func = piecewise({0.5: 1, 0.75: -1, 1: 0})
      >>> func(0.2)
      [0]
      >>> func(0.58)
      [1]

      >>> func = piecewise({0.5: [1, 0], 0.75: [0, 1]})
      >>> func(0.2)
      [0,0]
      >>> func(0.58)
      [1,0]
      >>> func(100)
      [0,1]

      >>> import math
      >>> func = piecewise({0: math.sin, 0.5: math.cos})
      >>> func(0.499)
      [0.47854771647582706]
      >>> func(0.5)
      [0.8775825618903728]

    """

    # first, sort the data (to simplify finding the right element
    # when calling the function)
    output_length = None  # the dimensionality of the returned values
    for time in data:
        if not is_number(time):
            raise ValidationError("Keys must be times (floats or ints), not %r"
                                  % type(time).__name__, attr='data')

        # figure out the length of this item
        if callable(data[time]):
            length = np.asarray(data[time](0.0)).size
        else:
            data[time] = np.asarray(data[time])
            length = data[time].size

        # make sure this is the same length as previous items
        if length != output_length and output_length is not None:
            raise ValidationError("time %g has %d items instead of %d" %
                                  (time, length, output_length), attr='data')
        output_length = length

    # make a default output of 0 when t before what was passed
    data[np.finfo(float).min] = np.zeros(output_length)
    ordered_data = OrderedDict(sorted(iteritems(data)))

    # build the function to return
    def piecewise_function(t, data=ordered_data):
        # get the t we'll use for output
        for time in (time for time in data if time <= t):
            out_t = time

        # if it's a function, call it
        if callable(data[out_t]):
            return np.asarray(data[out_t](t))
        return data[out_t]
    return piecewise_function


class HilbertCurve(object):
    """Hilbert curve function.

    Pre-calculates the Hilbert space filling curve with a given number
    of iterations. The curve will lie in the square delimited by the
    points (0, 0) and (1, 1).

    Arguments
    ---------
    n : int
        Iterations.
    """
    # Implementation based on
    # https://en.wikipedia.org/w/index.php?title=Hilbert_curve&oldid=633637210

    def __init__(self, n):
        self.n = n
        self.n_corners = (2 ** n) ** 2
        self.corners = np.zeros((self.n_corners, 2))
        self.steps = np.arange(self.n_corners)

        steps = np.arange(self.n_corners)
        for s in 2 ** np.arange(n):
            r = np.empty_like(self.corners, dtype='int')
            r[:, 0] = 1 & (steps // 2)
            r[:, 1] = 1 & (steps ^ r[:, 0])
            self._rot(s, r)
            self.corners += s * r
            steps //= 4

        self.corners /= (2 ** n) - 1

    def _rot(self, s, r):
        swap = r[:, 1] == 0
        flip = np.all(r == np.array([1, 0]), axis=1)

        self.corners[flip] = (s - 1 - self.corners[flip])
        self.corners[swap] = self.corners[swap, ::-1]

    def __call__(self, u):
        """Evaluate pre-calculated Hilbert curve.

        Arguments
        ---------
        u : ndarray (M,)
            Positions to evaluate on the curve in the range [0, 1].

        Returns
        -------
        ndarray (M, 2)
            Two-dimensional curve coordinates.
        """
        step = np.asarray(u * len(self.steps))
        return np.vstack((
            np.interp(step, self.steps, self.corners[:, 0]),
            np.interp(step, self.steps, self.corners[:, 1]))).T
