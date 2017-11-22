"""
These are borrowed from SciPy and used under their license:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2012 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""
# From scipy/sparse/sputils.py v0.18.0

from __future__ import absolute_import, division

import numpy as np


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    if issequence(x):
        return False
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False


def isshape(x):
    """Is x a valid 2-tuple of dimensions?
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if np.ndim(M) == 0 and np.ndim(N) == 0:
                return True
        return False


def issequence(t):
    return ((isinstance(t, (list, tuple)) and
            (len(t) == 0 or np.isscalar(t[0]))) or
            (isinstance(t, np.ndarray) and (t.ndim == 1)))


def ismatrix(t):
    return ((isinstance(t, (list, tuple)) and
             len(t) > 0 and issequence(t[0])) or
            (isinstance(t, np.ndarray) and t.ndim == 2))


def isdense(x):
    return isinstance(x, np.ndarray)
