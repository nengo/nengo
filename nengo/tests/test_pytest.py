import nengo.utils.numpy as npext
from nengo.tests.conftest import test_seed


def test_seed_fixture(seed):
    """The seed should be the same on all machines"""
    i = (seed - test_seed) % npext.maxint
    assert i == 1832276344
