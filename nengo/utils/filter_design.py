"""Functions for filter design.

----

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

import warnings

import numpy as np
from numpy import (
    product,
    zeros,
    array,
    dot,
    r_,
    eye,
    atleast_1d,
    atleast_2d,
    poly,
    roots,
    asarray,
    allclose,
)

from nengo._vendor.scipy import expm


class BadCoefficients(UserWarning):
    pass


def tf2zpk(b, a):
    """Return zero, pole, gain (z,p,k) representation from a numerator,
    denominator representation of a linear filter.

    Parameters
    ----------
    b : ndarray
        Numerator polynomial.
    a : ndarray
        Denominator polynomial.

    Returns
    -------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Notes
    -----
    If some values of `b` are too close to 0, they are removed. In that case,
    a BadCoefficients warning is emitted.

    """
    b, a = normalize(b, a)
    b = (b + 0.0) / a[0]
    a = (a + 0.0) / a[0]
    k = b[0]
    b /= b[0]
    z = roots(b)
    p = roots(a)
    return z, p, k


def zpk2tf(z, p, k):
    """Return polynomial transfer function representation from zeros
    and poles

    Parameters
    ----------
    z : ndarray
        Zeros of the transfer function.
    p : ndarray
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial.
    a : ndarray
        Denominator polynomial.

    """
    z = atleast_1d(z)
    k = atleast_1d(k)
    if len(z.shape) > 1:
        temp = poly(z[0])
        b = zeros((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * poly(z[i])
    else:
        b = k * poly(z)
    a = atleast_1d(poly(p))
    return b, a


def normalize(b, a):
    """Normalize polynomial representation of a transfer function.

    If values of `b` are too close to 0, they are removed. In that case, a
    BadCoefficients warning is emitted.

    """
    b, a = map(atleast_1d, (b, a))
    if len(a.shape) != 1:
        raise ValueError("Denominator polynomial must be rank-1 array.")
    if len(b.shape) > 2:
        raise ValueError("Numerator polynomial must be rank-1 or" " rank-2 array.")
    if len(b.shape) == 1:
        b = asarray([b], b.dtype.char)
    while a[0] == 0.0 and len(a) > 1:
        a = a[1:]
    outb = b * (1.0) / a[0]
    outa = a * (1.0) / a[0]
    if allclose(0, outb[:, 0], atol=1e-14):
        warnings.warn(
            "Badly conditioned filter coefficients (numerator): the "
            "results may be meaningless",
            BadCoefficients,
        )
        while allclose(0, outb[:, 0], atol=1e-14) and (outb.shape[-1] > 1):
            outb = outb[:, 1:]
    if outb.shape[0] == 1:
        outb = outb[0]
    return outb, outa


def tf2ss(num, den):
    """Transfer function to state-space representation.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the numerator and denominator polynomials.
        The denominator needs to be at least as long as the numerator.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system.

    """
    # Controller canonical state-space representation.
    #  if M+1 = len(num) and K+1 = len(den) then we must have M <= K
    #  states are found by asserting that X(s) = U(s) / D(s)
    #  then Y(s) = N(s) * X(s)
    #
    #   A, B, C, and D follow quite naturally.
    #
    num, den = normalize(num, den)  # Strips zeros, checks arrays
    nn = len(num.shape)
    if nn == 1:
        num = asarray([num], num.dtype)
    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # Null system
        return array([], float), array([], float), array([], float), array([], float)

    # pad numerator to have same number of columns has denominator
    num = r_["-1", zeros((num.shape[0], K - M), num.dtype), num]

    if num.shape[-1] > 0:
        D = num[:, 0]
    else:
        D = array([], float)

    if K == 1:
        return array([], float), array([], float), array([], float), D

    frow = -array([den[1:]])
    A = r_[frow, eye(K - 2, K - 1)]
    B = eye(K - 1, 1)
    C = num[:, 1:] - num[:, 0] * den[1:]
    return A, B, C, D


def _none_to_empty_2d(arg):
    if arg is None:
        return zeros((0, 0))
    else:
        return arg


def _atleast_2d_or_none(arg):
    if arg is not None:
        return atleast_2d(arg)


def _shape_or_none(M):
    if M is not None:
        return M.shape
    else:
        return (None,) * 2


def _choice_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg


def _restore(M, shape):
    if M.shape == (0, 0):
        return zeros(shape)
    else:
        if M.shape != shape:
            raise ValueError("The input arrays have incompatible shapes.")
        return M


def abcd_normalize(A=None, B=None, C=None, D=None):
    """Check state-space matrices and ensure they are rank-2.

    If enough information on the system is provided, that is, enough
    properly-shaped arrays are passed to the function, the missing ones
    are built from this information, ensuring the correct number of
    rows and columns. Otherwise a ValueError is raised.

    Parameters
    ----------
    A, B, C, D : array_like, optional
        State-space matrices. All of them are None (missing) by default.

    Returns
    -------
    A, B, C, D : array
        Properly shaped state-space matrices.

    Raises
    ------
    ValueError
        If not enough information on the system was provided.

    """
    A, B, C, D = map(_atleast_2d_or_none, (A, B, C, D))

    Am, _ = _shape_or_none(A)
    Bm, Bn = _shape_or_none(B)
    Cm, Cn = _shape_or_none(C)
    Dm, Dn = _shape_or_none(D)

    p = _choice_not_none(Am, Bm, Cn)
    q = _choice_not_none(Bn, Dn)
    r = _choice_not_none(Cm, Dm)
    if p is None or q is None or r is None:
        raise ValueError("Not enough information on the system.")

    A, B, C, D = map(_none_to_empty_2d, (A, B, C, D))
    A = _restore(A, (p, p))
    B = _restore(B, (p, q))
    C = _restore(C, (r, p))
    D = _restore(D, (r, q))

    return A, B, C, D


def ss2tf(A, B, C, D, input=0):
    """State-space to transfer function.

    Parameters
    ----------
    A, B, C, D : ndarray
        State-space representation of linear system.
    input : int, optional
        For multiple-input systems, the input to use.

    Returns
    -------
    num : 2-D ndarray
        Numerator(s) of the resulting transfer function(s).  `num` has one row
        for each of the system's outputs. Each row is a sequence representation
        of the numerator polynomial.
    den : 1-D ndarray
        Denominator of the resulting transfer function(s).  `den` is a sequence
        representation of the denominator polynomial.

    """
    # transfer function is C (sI - A)**(-1) B + D
    A, B, C, D = map(asarray, (A, B, C, D))
    # Check consistency and make them all rank-2 arrays
    A, B, C, D = abcd_normalize(A, B, C, D)

    nout, nin = D.shape
    if input >= nin:
        raise ValueError("System does not have the input specified.")

    # make MOSI from possibly MOMI system.
    if B.shape[-1] != 0:
        B = B[:, input]
    B.shape = (B.shape[0], 1)
    if D.shape[-1] != 0:
        D = D[:, input]

    try:
        den = poly(A)
    except ValueError:
        den = 1

    if (product(B.shape, axis=0) == 0) and (product(C.shape, axis=0) == 0):
        num = np.ravel(D)
        if (product(D.shape, axis=0) == 0) and (product(A.shape, axis=0) == 0):
            den = []
        return num, den

    num_states = A.shape[0]
    type_test = A[:, 0] + B[:, 0] + C[0, :] + D
    num = np.zeros((nout, num_states + 1), type_test.dtype)
    for k in range(nout):
        Ck = atleast_2d(C[k, :])
        num[k] = poly(A - dot(B, Ck)) + (D[k] - 1) * den

    return num, den


def zpk2ss(z, p, k):
    """Zero-pole-gain representation to state-space representation

    Parameters
    ----------
    z, p : sequence
        Zeros and poles.
    k : float
        System gain.

    Returns
    -------
    A, B, C, D : ndarray
        State-space matrices.

    """
    return tf2ss(*zpk2tf(z, p, k))


def ss2zpk(A, B, C, D, input=0):
    """State-space representation to zero-pole-gain representation.

    Parameters
    ----------
    A, B, C, D : ndarray
        State-space representation of linear system.
    input : int, optional
        For multiple-input systems, the input to use.

    Returns
    -------
    z, p : sequence
        Zeros and poles.
    k : float
        System gain.

    """
    return tf2zpk(*ss2tf(A, B, C, D, input=input))


def cont2discrete(sys, dt, method="zoh", alpha=None):  # noqa: C901
    """
    Transform a continuous to a discrete state-space system.

    Parameters
    ----------
    sys : a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

           * 2: (num, den)
           * 3: (zeros, poles, gain)
           * 4: (A, B, C, D)

    dt : float
        The discretization time step.
    method : {"gbt", "bilinear", "euler", "backward_diff", "zoh"}
        Which method to use:

           * gbt: generalized bilinear transformation
           * bilinear: Tustin's approximation ("gbt" with alpha=0.5)
           * euler: Euler (or forward differencing) method ("gbt" with alpha=0)
           * backward_diff: Backwards differencing ("gbt" with alpha=1.0)
           * zoh: zero-order hold (default)

    alpha : float within [0, 1]
        The generalized bilinear transformation weighting parameter, which
        should only be specified with method="gbt", and is ignored otherwise

    Returns
    -------
    sysd : tuple containing the discrete system
        Based on the input type, the output will be of the form

        * (num, den, dt)   for transfer function input
        * (zeros, poles, gain, dt)   for zeros-poles-gain input
        * (A, B, C, D, dt) for state-space system input

    Notes
    -----
    By default, the routine uses a Zero-Order Hold (zoh) method to perform
    the transformation.  Alternatively, a generalized bilinear transformation
    may be used, which includes the common Tustin's bilinear approximation,
    an Euler's method technique, or a backwards differencing technique.

    The Zero-Order Hold (zoh) method is based on [1]_, the generalized bilinear
    approximation is based on [2]_ and [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/
               Discretization#Discretization_of_linear_state_space_models

    .. [2] http://techteach.no/publications/
               discretetime_signals_systems/discrete.pdf

    .. [3] G. Zhang, X. Chen, and T. Chen, Digital redesign via the generalized
        bilinear transformation, Int. J. Control, vol. 82, no. 4, pp. 741-754,
        2009.
        (http://www.ece.ualberta.ca/~gfzhang/research/ZCC07_preprint.pdf)

    """
    if len(sys) == 2:
        sysd = cont2discrete(tf2ss(sys[0], sys[1]), dt, method=method, alpha=alpha)
        return ss2tf(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(sys) == 3:
        sysd = cont2discrete(
            zpk2ss(sys[0], sys[1], sys[2]), dt, method=method, alpha=alpha
        )
        return ss2zpk(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(sys) == 4:
        a, b, c, d = sys
    else:
        raise ValueError(
            "First argument must either be a tuple of 2 (tf), "
            "3 (zpk), or 4 (ss) arrays."
        )

    if method == "gbt":
        if alpha is None:
            raise ValueError(
                "Alpha parameter must be specified for the "
                "generalized bilinear transform (gbt) method"
            )
        elif alpha < 0 or alpha > 1:
            raise ValueError(
                "Alpha parameter must be within the interval "
                "[0,1] for the gbt method"
            )

    if method == "gbt":
        # This parameter is used repeatedly - compute once here
        ima = np.eye(a.shape[0]) - alpha * dt * a
        ad = np.linalg.solve(ima, np.eye(a.shape[0]) + (1.0 - alpha) * dt * a)
        bd = np.linalg.solve(ima, dt * b)

        # Similarly solve for the output equation matrices
        cd = np.linalg.solve(ima.transpose(), c.transpose())
        cd = cd.transpose()
        dd = d + alpha * np.dot(c, bd)

    elif method == "bilinear" or method == "tustin":
        return cont2discrete(sys, dt, method="gbt", alpha=0.5)

    elif method == "euler" or method == "forward_diff":
        return cont2discrete(sys, dt, method="gbt", alpha=0.0)

    elif method == "backward_diff":
        return cont2discrete(sys, dt, method="gbt", alpha=1.0)

    elif method == "zoh":
        # Build an exponential matrix
        em_upper = np.hstack((a, b))

        # Need to stack zeros under the a and b matrices
        em_lower = np.hstack(
            (np.zeros((b.shape[1], a.shape[0])), np.zeros((b.shape[1], b.shape[1])))
        )

        em = np.vstack((em_upper, em_lower))
        ms = expm(dt * em)

        # Dispose of the lower rows
        ms = ms[: a.shape[0], :]

        ad = ms[:, 0 : a.shape[1]]
        bd = ms[:, a.shape[1] :]

        cd = c
        dd = d

    else:
        raise ValueError("Unknown transformation method '%s'" % method)

    return ad, bd, cd, dd, dt
