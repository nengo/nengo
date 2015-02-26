import nengo.utils.numpy as npext
from nengo.conftest import TestConfig


def test_seed_fixture(seed):
    """The seed should be the same on all machines"""
    i = (seed - TestConfig.test_seed) % npext.maxint
    assert i == 1832276344
