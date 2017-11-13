import errno
import multiprocessing
import os
import timeit
from subprocess import CalledProcessError

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
from nengo.cache import (CacheIndex, DecoderCache, Fingerprint,
                         get_fragment_size, WriteableCacheIndex)
from nengo.exceptions import CacheIOWarning, FingerprintError
from nengo.solvers import LstsqL2
from nengo.utils.compat import int_types


class SolverMock(object):
    n_calls = {}

    def __init__(self):
        self.n_calls[self] = 0

    def __call__(self, conn, gain, bias, x, targets, rng=np.random, E=None):
        self.n_calls[self] += 1
        if E is None:
            return np.random.rand(x.shape[1], targets.shape[1]), {'info': 'v'}
        else:
            return np.random.rand(x.shape[1], E.shape[1]), {'info': 'v'}


def get_solver_test_args(**kwargs):
    M = 100
    N = 10
    D = 2
    conn = nengo.Connection(
        nengo.Ensemble(N, D, add_to_container=False),
        nengo.Node(size_in=D, add_to_container=False),
        add_to_container=False)
    conn.solver = kwargs.pop('solver', nengo.solvers.LstsqL2nz())
    defaults = {
        'conn': conn,
        'gain': np.ones(N),
        'bias': np.ones(N),
        'x': np.ones((M, D)),
        'targets': np.ones((M, N)),
        'rng': np.random.RandomState(42),
    }
    defaults.update(kwargs)
    return defaults


def get_weight_solver_test_args():
    M = 100
    N = 10
    N2 = 5
    D = 2
    conn = nengo.Connection(
        nengo.Ensemble(N, D, add_to_container=False),
        nengo.Node(size_in=D, add_to_container=False),
        solver=nengo.solvers.LstsqL2nz(), add_to_container=False)
    return {
        'conn': conn,
        'gain': np.ones(N),
        'bias': np.ones(N),
        'x': np.ones((M, D)),
        'targets': np.ones((M, N)),
        'rng': np.random.RandomState(42),
        'E': np.ones((D, N2)),
    }


def test_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    # Basic test, that results are cached.
    with DecoderCache(cache_dir=cache_dir) as cache:
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
        solver_args['gain'] *= 2
        decoders3, solver_info3 = cache.wrap_solver(solver_mock)(**solver_args)
        assert SolverMock.n_calls[solver_mock] == 2
        assert np.any(decoders1 != decoders3)

        # Test that the cache does not load results of another solver.
        another_solver = SolverMock()
        cache.wrap_solver(another_solver)(**get_solver_test_args(
            solver=nengo.solvers.LstsqNoise()))
        assert SolverMock.n_calls[another_solver] == 1


def test_corrupted_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    with DecoderCache(cache_dir=cache_dir) as cache:
        solver_mock = SolverMock()
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 1

        # corrupt the cache
        for path in cache.get_files():
            with open(path, 'w') as f:
                f.write('corrupted')

        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 2


def test_corrupted_decoder_cache_index(tmpdir):
    cache_dir = str(tmpdir)

    with DecoderCache(cache_dir=cache_dir):
        pass  # Initialize cache with required files
    assert len(os.listdir(cache_dir)) == 2  # index, index.lock

    # Write corrupted index
    with open(os.path.join(cache_dir, CacheIndex._INDEX), 'w') as f:
        f.write('(d')  # empty dict, but missing '.' at the end

    # Try to load index
    with DecoderCache(cache_dir=cache_dir):
        pass
    assert len(os.listdir(cache_dir)) == 2  # index, index.lock


def test_decoder_cache_invalidation(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    # Basic test, that results are cached.
    with DecoderCache(cache_dir=cache_dir) as cache:
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 1
        cache.invalidate()
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 2


def test_decoder_cache_size_includes_overhead(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    with DecoderCache(cache_dir=cache_dir) as cache:
        cache.wrap_solver(solver_mock)(**get_solver_test_args())

        fragment_size = get_fragment_size(cache_dir)
        actual_size = sum(os.stat(p).st_size for p in cache.get_files())
        assert actual_size % fragment_size != 0, (
            'Test succeeded by chance. Adjust get_solver_test_args() to '
            'produce date not aligned with the files system fragment size.')

        assert cache.get_size_in_bytes() % fragment_size == 0


def test_decoder_cache_shrinking(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()
    another_solver = SolverMock()

    with DecoderCache(cache_dir=cache_dir) as cache:
        cache.wrap_solver(solver_mock)(**get_solver_test_args())

        # Ensure differing time stamps (depending on the file system the
        # timestamp resolution might be as bad as 1 day).
        for path in cache.get_files():
            timestamp = os.stat(path).st_atime
            timestamp -= 60 * 60 * 24 * 2  # 2 days
            os.utime(path, (timestamp, timestamp))

    with DecoderCache(cache_dir=cache_dir) as cache:
        cache.wrap_solver(another_solver)(**get_solver_test_args(
            solver=nengo.solvers.LstsqNoise()))

        cache_size = cache.get_size_in_bytes()
        assert cache_size > 0

        cache.shrink(cache_size - 1)

        # check that older cached result was removed
        assert SolverMock.n_calls[solver_mock] == 1
        cache.wrap_solver(another_solver)(**get_solver_test_args(
            solver=nengo.solvers.LstsqNoise()))
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 2
        assert SolverMock.n_calls[another_solver] == 1


def test_decoder_cache_shrink_threadsafe(monkeypatch, tmpdir):
    """Tests that shrink handles files deleted by other processes."""
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    with DecoderCache(cache_dir=cache_dir) as cache:
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        limit = cache.get_size()

        # Ensure differing time stamps (depending on the file system the
        # timestamp resolution might be as bad as 1 day).
        for filename in os.listdir(cache.cache_dir):
            path = os.path.join(cache.cache_dir, filename)
            timestamp = os.stat(path).st_atime
            timestamp -= 60 * 60 * 24 * 2  # 2 days
            os.utime(path, (timestamp, timestamp))

        cache.wrap_solver(solver_mock)(**get_solver_test_args(
            solver=nengo.solvers.LstsqNoise()))

        cache_size = cache.get_size_in_bytes()
        assert cache_size > 0

        def raise_file_not_found(orig_fn):
            def fn(filename, *args, **kwargs):
                if filename.endswith('.lock'):
                    return orig_fn(filename, *args, **kwargs)
                raise OSError(errno.ENOENT, "File not found.")
            return fn

        monkeypatch.setattr(cache, 'get_size_in_bytes', lambda: cache_size)
        monkeypatch.setattr('os.stat', raise_file_not_found(os.stat))
        monkeypatch.setattr('os.remove', raise_file_not_found(os.remove))
        monkeypatch.setattr('os.unlink', raise_file_not_found(os.unlink))

        cache.shrink(limit)


def test_decoder_cache_with_E_argument_to_solver(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    with DecoderCache(cache_dir=cache_dir) as cache:
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


nengo.cache.Fingerprint.whitelist(DummyA)


class DummyB(object):
    def __init__(self, attr=0):
        self.attr = attr


nengo.cache.Fingerprint.whitelist(DummyB)


def dummy_fn(arg):
    pass


@pytest.mark.parametrize('reference, equal, different', (
    (True, True, False),             # bool
    (False, False, True),            # bool
    (1.0, 1.0, 2.0),                 # float
    (1.0 + 2.0j, 1 + 2j, 2.0 + 1j),  # complex
    (b'a', b'a', b'b'),              # bytes
    ((0, 1), (0, 1), (0, 2)),        # tuple
    ([0, 1], [0, 1], [0, 2]),        # list
    (u'a', u'a', u'b'),              # unicode string
    (np.eye(2), np.eye(2), np.array([[0, 1], [1, 0]])),      # array
    (DummyA(), DummyA(), DummyB()),  # object instance
    (DummyA(1), DummyA(1), DummyA(2)),     # object instance
    (LstsqL2(reg=.1), LstsqL2(reg=.1), LstsqL2(reg=.2)),     # solver
) + tuple((typ(1), typ(1), typ(2)) for typ in int_types))
def test_fingerprinting(reference, equal, different):
    assert str(Fingerprint(reference)) == str(Fingerprint(equal))
    assert str(Fingerprint(reference)) != str(Fingerprint(different))


@pytest.mark.parametrize('obj', (
    np.array([object()]),   # array
    np.array([(1.,)], dtype=[('field1', 'f8')]),  # array
    {'a': 1, 'b': 2},       # dict
    object(),               # object instance
    dummy_fn,               # function
))
def test_unsupported_fingerprinting(obj):
    with pytest.raises(FingerprintError):
        Fingerprint(obj)


def test_fails_for_lambda_expression():
    with pytest.raises(FingerprintError):
        Fingerprint(lambda x: x)


def test_cache_works(tmpdir, RefSimulator, seed):
    cache_dir = str(tmpdir)

    model = nengo.Network(seed=seed)
    with model:
        nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    assert len(os.listdir(cache_dir)) == 0
    with RefSimulator(model, model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir))):
        assert len(os.listdir(cache_dir)) == 3  # index, index.lock, and *.nco


def test_cache_not_used_without_seed(tmpdir, RefSimulator):
    cache_dir = str(tmpdir)

    model = nengo.Network()
    with model:
        nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    assert len(os.listdir(cache_dir)) == 0
    with RefSimulator(model, model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir))):
        assert len(os.listdir(cache_dir)) == 2  # index, index.lock


def build_many_ensembles(cache_dir, RefSimulator):
    with nengo.Network(seed=1) as model:
        for _ in range(100):
            nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    with RefSimulator(model, model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir))):
        pass


@pytest.mark.slow
def test_cache_concurrency(tmpdir, RefSimulator):
    cache_dir = str(tmpdir)

    n_processes = 100
    processes = [
        multiprocessing.Process(
            target=build_many_ensembles, args=(cache_dir, RefSimulator))
        for _ in range(n_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join(60)
    for p in processes:
        assert p.exitcode == 0


def reject_outliers(data):
    med = np.median(data)
    limits = 1.5 * (np.percentile(data, [25, 75]) - med) + med
    return np.asarray(data)[np.logical_and(data > limits[0], data < limits[1])]


class TestCacheBenchmark(object):
    n_trials = 25

    setup = '''
import numpy as np
import nengo
import nengo.cache
from nengo.rc import rc

rc.set("decoder_cache", "path", {tmpdir!r})

model = nengo.Network(seed=1)
with model:
    a = nengo.Ensemble({N}, dimensions={D}, n_eval_points={M})
    b = nengo.Ensemble({N}, dimensions={D}, n_eval_points={M})
    conn = nengo.Connection(a, b)
    '''

    without_cache = {
        'rc': 'rc.set("decoder_cache", "enabled", "False")',
        'stmt': '''
with nengo.Simulator(model):
    pass
'''
    }

    with_cache_miss_ro = {
        'rc': '''
with nengo.cache.DecoderCache() as cache:
    cache.invalidate()
rc.set("decoder_cache", "enabled", "True")
rc.set("decoder_cache", "readonly", "True")
''',
        'stmt': '''
with nengo.Simulator(model):
    pass
'''
    }

    with_cache_miss = {
        'rc': '''
with nengo.cache.DecoderCache() as cache:
    cache.invalidate()
rc.set("decoder_cache", "enabled", "True")
rc.set("decoder_cache", "readonly", "False")
''',
        'stmt': '''
with nengo.Simulator(model):
    pass
'''
    }

    with_cache_hit = {
        'rc': '''
rc.set("decoder_cache", "enabled", "True")
rc.set("decoder_cache", "readonly", "False")
with nengo.Simulator(model):
    pass
''',
        'stmt': '''
with nengo.Simulator(model):
    pass
'''
    }

    labels = ["no cache", "cache miss", "cache miss ro", "cache hit"]
    keys = [l.replace(' ', '_') for l in labels]
    param_to_axis_label = {
        'D': "dimensions",
        'N': "neurons",
        'M': "evaluation points"
    }
    defaults = {'D': 1, 'N': 50, 'M': 1000}

    def time_code(self, code, **kwargs):
        return timeit.repeat(
            stmt=code['stmt'], setup=self.setup.format(**kwargs) + code['rc'],
            number=1, repeat=self.n_trials)

    def time_all(self, **kwargs):
        return (self.time_code(self.without_cache, **kwargs),
                self.time_code(self.with_cache_miss, **kwargs),
                self.time_code(self.with_cache_miss_ro, **kwargs),
                self.time_code(self.with_cache_hit, **kwargs))

    def get_args(self, varying_param, value):
        args = dict(self.defaults)  # make a copy
        args[varying_param] = value
        return args

    @pytest.mark.slow
    @pytest.mark.noassertions
    @pytest.mark.parametrize('varying_param', ['D', 'N', 'M'])
    def test_cache_benchmark(self, tmpdir, varying_param, analytics, plt):
        varying = {
            'D': np.asarray(np.linspace(1, 512, 10), dtype=int),
            'N': np.asarray(np.linspace(10, 500, 8), dtype=int),
            'M': np.asarray(np.linspace(750, 2500, 8), dtype=int)
        }[varying_param]
        axis_label = self.param_to_axis_label[varying_param]

        times = [
            self.time_all(
                tmpdir=str(tmpdir), **self.get_args(varying_param, v))
            for v in varying]

        for i, data in enumerate(zip(*times)):
            plt.plot(varying, np.median(data, axis=1), label=self.labels[i])
            plt.xlim(right=varying[-1])
            analytics.add_data(varying_param, varying, axis_label)
            analytics.add_data(self.keys[i], data)

        plt.xlabel("Number of %s" % axis_label)
        plt.ylabel("Build time (s)")
        plt.legend(loc='best')

    @pytest.mark.compare
    @pytest.mark.parametrize('varying_param', ['D', 'N', 'M'])
    def test_compare_cache_benchmark(
            self, varying_param, analytics_data, plt, logger):
        stats = pytest.importorskip('scipy.stats')

        d1, d2 = analytics_data
        assert np.all(d1[varying_param] == d2[varying_param]), (
            'Cannot compare different parametrizations')
        axis_label = self.param_to_axis_label[varying_param]

        logger.info("Cache, varying %s:", axis_label)
        for label, key in zip(self.labels, self.keys):
            clean_d1 = [reject_outliers(d) for d in d1[key]]
            clean_d2 = [reject_outliers(d) for d in d2[key]]
            diff = [np.median(b) - np.median(a)
                    for a, b in zip(clean_d1, clean_d2)]

            p_values = np.array(
                [2. * stats.mannwhitneyu(a, b, alternative='two-sided')[1]
                 for a, b in zip(clean_d1, clean_d2)])
            overall_p = 1. - np.prod(1. - p_values)
            if overall_p < .05:
                logger.info("  %s: Significant change (p <= %.3f). See plots.",
                            label, np.ceil(overall_p * 1000.) / 1000.)
            else:
                logger.info("  %s: No significant change.", label)

            plt.plot(d1[varying_param], diff, label=label)

        plt.xlabel("Number of %s" % axis_label)
        plt.ylabel("Difference in build time (s)")
        plt.legend(loc='best')


class TestCacheShrinkBenchmark(object):
    n_trials = 50

    setup = '''
import numpy as np
import nengo
import nengo.cache
from nengo.rc import rc

rc.set("decoder_cache", "path", {tmpdir!r})

for i in range(10):
    model = nengo.Network(seed=i)
    with model:
        a = nengo.networks.EnsembleArray(10, 128, 1)
        b = nengo.networks.EnsembleArray(10, 128, 1)
        conn = nengo.Connection(a.output, b.input)
    with nengo.Simulator(model):
        pass

rc.set("decoder_cache", "size", "0KB")
cache = nengo.cache.DecoderCache()
    '''

    stmt = 'with cache: cache.shrink()'

    @pytest.mark.slow
    @pytest.mark.noassertions
    def test_cache_shrink_benchmark(self, tmpdir, analytics, logger):
        times = timeit.repeat(
            stmt=self.stmt, setup=self.setup.format(tmpdir=str(tmpdir)),
            number=1, repeat=self.n_trials)
        logger.info("Shrink took a minimum of %f seconds.", np.min(times))
        logger.info("Shrink took a %f seconds on average.", np.mean(times))
        logger.info(
            "Shrink took a %f seconds on average with outliers rejected.",
            np.mean(reject_outliers(times)))
        analytics.add_data('times', times)

    @pytest.mark.compare
    def test_compare_cache_shrink_benchmark(self, analytics_data, plt, logger):
        stats = pytest.importorskip('scipy.stats')

        d1, d2 = (x['times'] for x in analytics_data)
        clean_d1 = reject_outliers(d1)
        clean_d2 = reject_outliers(d2)

        diff = np.median(clean_d2) - np.median(clean_d1)

        p_value = 2. * stats.mannwhitneyu(
            clean_d1, clean_d2, alternative='two-sided')[1]
        if p_value < .05:
            logger.info("Significant change of %d seconds (p <= %.3f).",
                        diff, np.ceil(p_value * 1000.) / 1000.)
        else:
            logger.info("No significant change (%d).", diff)
        logger.info("Speed up: %s", np.median(clean_d1) / np.median(clean_d2))

        plt.scatter(np.ones_like(d1), d1, c='b')
        plt.scatter(2 * np.ones_like(d2), d2, c='g')


def test_warns_out_of_context(tmpdir):
    cache_dir = str(tmpdir)
    cache = DecoderCache(cache_dir=cache_dir)

    solver_mock = SolverMock()
    solver = cache.wrap_solver(solver_mock)
    with pytest.warns(UserWarning):
        solver(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1


def test_cacheindex_cannot_write(tmpdir):
    index = WriteableCacheIndex(cache_dir=str(tmpdir))
    with index:
        index[0] = ("file0", 0, 0)
    mtime = os.stat(index.index_path).st_mtime

    index = CacheIndex(cache_dir=str(tmpdir))
    with index:
        with pytest.raises(TypeError):
            index[0] = ("file", 0, 0)
        with pytest.raises(TypeError):
            del index[0]
        assert index[0] == ("file0", 0, 0)
    assert os.stat(index.index_path).st_mtime == mtime


def test_writeablecacheindex_writes(tmpdir):
    index = WriteableCacheIndex(cache_dir=str(tmpdir))
    with index:
        index[0] = ("file0", 0, 0)
        index[1] = ("file1", 0, 0)
        del index[1]

    # Verify with readonly cacheindex
    index = CacheIndex(cache_dir=str(tmpdir))
    with index:
        assert index[0] == ("file0", 0, 0)
        assert 1 not in index


def test_writeablecacheindex_setitem(tmpdir):
    index = WriteableCacheIndex(cache_dir=str(tmpdir))

    with pytest.raises(ValueError):
        index[0] = "file0"
    with pytest.raises(ValueError):
        index[0] = ("file0", 0)
    with pytest.raises(ValueError):
        index[0] = ("file0", 0, 0, 0)


def test_writeablecacheindex_removes(tmpdir):
    index = WriteableCacheIndex(cache_dir=str(tmpdir))
    with index:
        index[0] = ("file0", 0, 0)
        index[1] = ("file1", 0, 0)
        index.remove_file_entry("file0")
        index.remove_file_entry(os.path.join(str(tmpdir), "file1"))

    # Verify with readonly cacheindex
    index = CacheIndex(cache_dir=str(tmpdir))
    with index:
        assert 0 not in index, "Fails on relative paths"
        assert 1 not in index, "Fails on absolute paths"


def test_writeablecacheindex_warning(monkeypatch, tmpdir):

    def raise_error(*args, **kwargs):
        raise CalledProcessError(-1, "move")

    monkeypatch.setattr(nengo.cache, "replace", raise_error)
    with pytest.warns(CacheIOWarning):
        with WriteableCacheIndex(cache_dir=str(tmpdir)):
            pass


def test_shrink_does_not_fail_if_lock_cannot_be_acquired(tmpdir):
    cache = DecoderCache(cache_dir=str(tmpdir))
    cache._index._lock.timeout = 1.
    with cache:
        cache.wrap_solver(SolverMock())(**get_solver_test_args())
    with cache._index._lock:
        cache.shrink(limit=0)
