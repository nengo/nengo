import os

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
from nengo.cache import DecoderCache, Fingerprint
from nengo.utils.compat import int_types


class SolverMock(object):
    n_calls = {}

    def __init__(self, name='solver_mock'):
        self.n_calls[self] = 0
        self.__module__ = __name__
        self.__name__ = name

    def __call__(self, A, Y, rng=np.random, E=None):
        self.n_calls[self] += 1
        if E is None:
            return np.random.rand(A.shape[1], Y.shape[1]), {'info': 'v'}
        else:
            return np.random.rand(A.shape[1], E.shape[1]), {'info': 'v'}


def test_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    solver_mock = SolverMock()
    decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
        activities, targets, rng)
    assert SolverMock.n_calls[solver_mock] == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
        activities, targets, rng)
    assert SolverMock.n_calls[solver_mock] == 1  # result read from cache?
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2

    decoders3, solver_info3 = cache.wrap_solver(solver_mock)(
        2 * activities, targets, rng)
    assert SolverMock.n_calls[solver_mock] == 2
    assert np.any(decoders1 != decoders3)

    # Test that the cache does not load results of another solver.
    another_solver = SolverMock('another_solver')
    cache.wrap_solver(another_solver)(activities, targets, rng)
    assert SolverMock.n_calls[another_solver] == 1


def test_decoder_cache_invalidation(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock)(activities, targets, rng)
    assert SolverMock.n_calls[solver_mock] == 1
    cache.invalidate()
    cache.wrap_solver(solver_mock)(activities, targets, rng)
    assert SolverMock.n_calls[solver_mock] == 2


def test_decoder_cache_shrinking(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()
    another_solver = SolverMock('another_solver')

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock)(activities, targets, rng)

    # Ensure differing time stamps (depending on the file system the timestamp
    # resolution might be as bad as 1 day).
    for filename in os.listdir(cache.cache_dir):
        path = os.path.join(cache.cache_dir, filename)
        timestamp = os.stat(path).st_atime
        timestamp -= 60 * 60 * 24 * 2  # 2 days
        os.utime(path, (timestamp, timestamp))

    cache.wrap_solver(another_solver)(activities, targets, rng)

    assert cache.get_size() > 0

    cache.shrink(1)

    # check that older cached result was removed
    assert SolverMock.n_calls[solver_mock] == 1
    cache.wrap_solver(another_solver)(activities, targets, rng)
    cache.wrap_solver(solver_mock)(activities, targets, rng)
    assert SolverMock.n_calls[solver_mock] == 2
    assert SolverMock.n_calls[another_solver] == 1


def test_decoder_cache_with_E_argument_to_solver(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    M = 100
    N = 10
    N2 = 5
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)
    E = np.ones((D, N2))

    cache = DecoderCache(cache_dir=cache_dir)
    decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
        activities, targets, rng, E=E)
    assert SolverMock.n_calls[solver_mock] == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
        activities, targets, rng, E=E)
    assert SolverMock.n_calls[solver_mock] == 1  # read from cache?
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2


class DummyA(object):
    def __init__(self, attr=0):
        self.attr = attr


class DummyB(object):
    def __init__(self, attr=0):
        self.attr = attr


def dummy_fn_a(arg):
    pass


def dummy_fn_b(arg):
    pass


@pytest.mark.parametrize('reference, equal, different', (
    (True, True, False),             # bool
    (False, False, True),            # bool
    (1.0, 1.0, 2.0),                 # float
    (1.0 + 2.0j, 1 + 2j, 2.0 + 1j),  # complex
    (b'a', b'a', b'b'),              # bytes
    (u'a', u'a', u'b'),              # unicode string
    (np.eye(2), np.eye(2), np.array([[0, 1], [1, 0]])),      # array
    ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}, {'a': 2, 'b': 1}),  # dict
    ((1, 2), (1, 2), (2, 1)),        # tuple
    ([1, 2], [1, 2], [2, 1]),        # list
    (DummyA(), DummyA(), DummyB()),  # object instance
    (DummyA(1), DummyA(1), DummyA(2)),     # object instance
    (DummyA(1), DummyA(1), DummyA(2)),     # object instance
    (dummy_fn_a, dummy_fn_a, dummy_fn_b),  # function
) + tuple((typ(1), typ(1), typ(2)) for typ in int_types))
def test_fingerprinting(reference, equal, different):
    assert str(Fingerprint(reference)) == str(Fingerprint(equal))
    assert str(Fingerprint(reference)) != str(Fingerprint(different))


def test_fails_for_lambda_expression():
    with pytest.raises(ValueError):
        Fingerprint(lambda x: x)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
