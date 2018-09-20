import nengo.utils.numpy as npext
from nengo.conftest import TestConfig


def test_seed_fixture(seed):
    """The seed should be the same on all machines"""
    i = (seed - TestConfig.test_seed) % npext.maxint
    assert i == 1832276344


def test_allclose(allclose):
    # if the nengo_test_tolerances in setup.cfg are working properly, then
    # atol will be set such that all these assertions pass
    assert allclose(0, 1.9, atol=0)
    assert allclose(0, 0.9, atol=0)
    assert allclose(0, 0.9, atol=0)
    assert not allclose(0, 1.9)


def test_unsupported():
    # if the nengo_test_unsupported config is working properly then this
    # test will be marked as xfail
    assert False
