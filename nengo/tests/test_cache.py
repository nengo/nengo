import errno
import os

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
from nengo.cache import DecoderCache, Fingerprint, NoDecoderCache
from nengo.utils.compat import int_types
from nengo.utils.testing import Timer


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


def get_solver_test_args():
    M = 100
    N = 10
    D = 2
    return {
        'activities': np.ones((M, D)),
        'targets': np.ones((M, N)),
        'rng': np.random.RandomState(42),
    }


def get_weight_solver_test_args():
    M = 100
    N = 10
    N2 = 5
    D = 2
    return {
        'activities': np.ones((M, D)),
        'targets': np.ones((M, N)),
        'rng': np.random.RandomState(42),
        'E': np.ones((D, N2)),
    }


def test_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    solver_mock = SolverMock()
    decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
        **get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
        **get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1  # result read from cache?
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2

    solver_args = get_solver_test_args()
    solver_args['activities'] *= 2
    decoders3, solver_info3 = cache.wrap_solver(solver_mock)(**solver_args)
    assert SolverMock.n_calls[solver_mock] == 2
    assert np.any(decoders1 != decoders3)

    # Test that the cache does not load results of another solver.
    another_solver = SolverMock('another_solver')
    cache.wrap_solver(another_solver)(**get_solver_test_args())
    assert SolverMock.n_calls[another_solver] == 1


def test_corrupted_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    cache = DecoderCache(cache_dir=cache_dir)
    solver_mock = SolverMock()
    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1

    # corrupt the cache
    for filename in os.listdir(cache_dir):
        with open(os.path.join(cache_dir, filename), 'w') as f:
            f.write('corrupted')

    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 2


def test_decoder_cache_invalidation(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1
    cache.invalidate()
    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 2


def test_decoder_cache_size_includes_overhead(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock)(**get_solver_test_args())

    fragment_size = os.statvfs(cache_dir).f_frsize
    actual_size = sum(os.stat(os.path.join(cache_dir, f)).st_size
                      for f in cache.get_files())
    assert actual_size % fragment_size != 0, (
        'Test succeeded by chance. Adjust get_solver_test_args() to produce '
        'date not aligned with the files system fragment size.')

    assert cache.get_size_in_bytes() % fragment_size == 0


def test_decoder_cache_shrinking(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()
    another_solver = SolverMock('another_solver')

    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    limit = cache.get_size()

    # Ensure differing time stamps (depending on the file system the timestamp
    # resolution might be as bad as 1 day).
    for filename in os.listdir(cache.cache_dir):
        path = os.path.join(cache.cache_dir, filename)
        timestamp = os.stat(path).st_atime
        timestamp -= 60 * 60 * 24 * 2  # 2 days
        os.utime(path, (timestamp, timestamp))

    cache.wrap_solver(another_solver)(**get_solver_test_args())

    assert cache.get_size_in_bytes() > 0

    cache.shrink(limit)

    # check that older cached result was removed
    assert SolverMock.n_calls[solver_mock] == 1
    cache.wrap_solver(another_solver)(**get_solver_test_args())
    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 2
    assert SolverMock.n_calls[another_solver] == 1


def test_decoder_cache_shrink_threadsafe(monkeypatch, tmpdir):
    """Tests that shrink handles files deleted by other processes."""
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()
    another_solver = SolverMock('another_solver')

    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock)(**get_solver_test_args())
    limit = cache.get_size()

    # Ensure differing time stamps (depending on the file system the timestamp
    # resolution might be as bad as 1 day).
    for filename in os.listdir(cache.cache_dir):
        path = os.path.join(cache.cache_dir, filename)
        timestamp = os.stat(path).st_atime
        timestamp -= 60 * 60 * 24 * 2  # 2 days
        os.utime(path, (timestamp, timestamp))

    cache.wrap_solver(another_solver)(**get_solver_test_args())

    cache_size = cache.get_size_in_bytes()
    assert cache_size > 0

    def raise_file_not_found(*args, **kwargs):
        raise OSError(errno.ENOENT, "File not found.")

    monkeypatch.setattr(cache, 'get_size_in_bytes', lambda: cache_size)
    monkeypatch.setattr('os.stat', raise_file_not_found)
    monkeypatch.setattr('os.remove', raise_file_not_found)
    monkeypatch.setattr('os.unlink', raise_file_not_found)

    cache.shrink(limit)


def test_decoder_cache_with_E_argument_to_solver(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    cache = DecoderCache(cache_dir=cache_dir)
    decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
        **get_weight_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
        **get_weight_solver_test_args())
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
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 2, 'b': 1}),  # dict
    ((1, 2), (1, 2), (2, 1)),        # tuple
    ([1, 2], [1, 2], [2, 1]),        # list
    (DummyA(), DummyA(), DummyB()),  # object instance
    (DummyA(1), DummyA(1), DummyA(2)),     # object instance
    (dummy_fn_a, dummy_fn_a, dummy_fn_b),  # function
) + tuple((typ(1), typ(1), typ(2)) for typ in int_types))
def test_fingerprinting(reference, equal, different):
    assert str(Fingerprint(reference)) == str(Fingerprint(equal))
    assert str(Fingerprint(reference)) != str(Fingerprint(different))


def test_fails_for_lambda_expression():
    with pytest.raises(ValueError):
        Fingerprint(lambda x: x)


def test_cache_works(tmpdir, Simulator, seed):
    cache_dir = str(tmpdir)

    model = nengo.Network(seed=seed)
    with model:
        nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    assert len(os.listdir(cache_dir)) == 0
    Simulator(model, model=nengo.builder.Model(
        dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir)))
    assert len(os.listdir(cache_dir)) == 1


def calc_relative_timer_diff(t1, t2):
    return (t2.duration - t1.duration) / (t2.duration + t1.duration)


@pytest.mark.benchmark
def test_cache_performance(tmpdir, Simulator, seed):
    cache_dir = str(tmpdir)

    model = nengo.Network(seed=seed)
    with model:
        nengo.Connection(nengo.Ensemble(2000, 10), nengo.Ensemble(2000, 10))

    with Timer() as t_no_cache:
        Simulator(model, model=nengo.builder.Model(
            dt=0.001, decoder_cache=NoDecoderCache()))
    with Timer() as t_cache_miss:
        Simulator(model, model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir)))
    with Timer() as t_cache_hit:
        Simulator(model, model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir)))

    assert calc_relative_timer_diff(t_no_cache, t_cache_miss) < 0.1
    assert calc_relative_timer_diff(t_cache_hit, t_no_cache) < 0.4


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
