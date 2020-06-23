import pytest

import numpy as np
from nengo.exceptions import ValidationError
from nengo.utils.least_squares_solvers import (
    Conjgrad,
    BlockConjgrad,
    SVD,
    RandomizedSVD,
)


def test_conjgrad_validationerror():
    """ensures that incorrect call throws validationerror"""
    with pytest.raises(ValidationError):
        con = Conjgrad(X0=([[1], [2], [3]]))
        A = np.array([[1], [2], [3]])
        Y = np.array([1, 2, 3])
        sigma = 0
        con(A, Y, sigma)


def test_conjgrad_iters():
    """tests _conjgrad_iters"""
    con = Conjgrad(X0=([[1], [2], [3]]))
    con._conjgrad_iters(np.sin, np.array([100]), 3, rtol=0)


def test_blockconjgrad_validationerror():
    """ensures that incorrect call throws validationerror"""
    with pytest.raises(ValidationError):
        con = BlockConjgrad(X0=([[1], [2], [3]]))
        A = np.array([[1], [2], [3]])
        Y = np.array([1, 2, 3])
        sigma = 0
        con(A, Y, sigma)


def test_svd():
    """tests SVD"""
    pytest.importorskip("sklearn")
    mySVD = SVD()
    A = np.array([[1], [2], [3]])
    Y = np.array([1, 2, 3])
    sigma = 0
    assert (
        repr(mySVD(A, Y, sigma)) == "(array([1.]), {'rmses': "
        "array([6.69210662e-16])})"
        or repr(mySVD(A, Y, sigma)) == "(array([1.]), {'rmses': "
        "array([0.])})"  # travis-ci version
    )


def test_randsvd():
    """tests RandomizedSVD"""
    pytest.importorskip("sklearn")
    mySVD = RandomizedSVD()
    A = np.array([[1], [2], [3]])
    Y = np.array([1, 2, 3])
    sigma = 0
    assert (
        repr(mySVD(A, Y, sigma)) == "(array([[1.]]), {'rmses': "
        "array([6.69210662e-16])})"
        or repr(mySVD(A, Y, sigma)) == "(array([[1.]]), {'rmses': "
        "array([0.])})"  # travis-ci version
    )
