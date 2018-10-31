"""
Extra functions to extend the capabilities of Numba.
"""

from __future__ import absolute_import

import numpy as np

from numba.extending import overload


@overload(np.clip)
def np_clip(a, a_min, a_max):
    """Numba-implementation of np.clip."""
    # Does not support `out` argument, optional arguments, nor a.clip
    # https://github.com/numba/numba/pull/3468
    def np_clip_impl(a, a_min, a_max):
        out = np.empty_like(a)
        for index, val in np.ndenumerate(a):
            if val < a_min:
                out[index] = a_min
            elif val > a_max:
                out[index] = a_max
            else:
                out[index] = val
        return out
    return np_clip_impl
