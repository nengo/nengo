import nengo.utils.numpy as npext
from nengo.conftest import TestConfig


def test_seed_fixture(seed):
    """The seed should be the same on all machines"""
    i = (seed - TestConfig.test_seed) % npext.maxint
    assert i == 1832276344


def test_allclose(allclose):
    assert allclose(0, 1.9, atol=0)
    assert allclose(0, 0.9, atol=0)
    assert allclose(0, 0.9, atol=0)
    assert not allclose(0, 1.9)
