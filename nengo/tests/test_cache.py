import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
from nengo.cache import DecoderCache


class DecoderSolverMock(object):
    def __init__(self, name='solver_mock'):
        self.n_calls = 0
        self.__module__ = __name__
        self.__name__ = name

    def get_solver_fn(self):
        def solver_fn(A, Y, rng=np.random, E=None):
            self.n_calls += 1
            if E is None:
                return np.random.rand(A.shape[1], Y.shape[1]), {'info': 'v'}
            else:
                return np.random.rand(A.shape[1], E.shape[1]), {'info': 'v'}
        solver_fn.__name__ = self.__name__
        return solver_fn


def test_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = DecoderSolverMock()

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    decoders1, solver_info1 = cache.wrap_solver(solver_mock.get_solver_fn())(
        activities, targets, rng)
    assert solver_mock.n_calls == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock.get_solver_fn())(
        activities, targets, rng)
    assert solver_mock.n_calls == 1  # check the result is read from cache
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2

    decoders3, solver_info3 = cache.wrap_solver(solver_mock.get_solver_fn())(
        2 * activities, targets, rng)
    assert solver_mock.n_calls == 2
    assert np.any(decoders1 != decoders3)

    # Test that the cache does not load results of another solver.
    another_solver = DecoderSolverMock('another_solver')
    cache.wrap_solver(another_solver.get_solver_fn())(activities, targets, rng)
    assert another_solver.n_calls == 1

    # Test get_size
    assert cache.get_size() > 0


def test_decoder_cache_with_E_argument_to_solver(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = DecoderSolverMock()

    M = 100
    N = 10
    N2 = 5
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)
    E = np.ones((D, N2))

    cache = DecoderCache(cache_dir=cache_dir)
    decoders1, solver_info1 = cache.wrap_solver(solver_mock.get_solver_fn())(
        activities, targets, rng, E=E)
    assert solver_mock.n_calls == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock.get_solver_fn())(
        activities, targets, rng, E=E)
    assert solver_mock.n_calls == 1  # check the result is read from cache
    assert_equal(decoders1, decoders2)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
