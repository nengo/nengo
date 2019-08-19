import errno
import multiprocessing
import os
import sys
from subprocess import CalledProcessError

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
from nengo.cache import (
    CacheIndex,
    DecoderCache,
    Fingerprint,
    get_fragment_size,
    WriteableCacheIndex,
)
from nengo.exceptions import CacheIOWarning, FingerprintError
import nengo.neurons
from nengo.solvers import LstsqL2
import nengo.utils.least_squares_solvers
from nengo.utils.testing import Mock


def isstrictsubclass(x, cls):
    return isinstance(x, type) and issubclass(x, cls) and x is not cls


def list_objects(module):
    return [getattr(module, key) for key in dir(module) if not key.startswith("_")]


neuron_types = [
    obj
    for obj in list_objects(nengo.neurons)
    if isstrictsubclass(obj, nengo.neurons.NeuronType)
]
solver_types = [
    obj
    for obj in list_objects(nengo.solvers)
    if isstrictsubclass(obj, nengo.solvers.Solver)
    and obj is not nengo.solvers.NoSolver  # NoSolver is not cacheable
]
lstsq_solver_types = [
    obj
    for obj in list_objects(nengo.utils.least_squares_solvers)
    if isstrictsubclass(obj, nengo.utils.least_squares_solvers.LeastSquaresSolver)
]


class SolverMock:
    n_calls = {}

    def __init__(self):
        self.n_calls[self] = 0

    def __call__(self, conn, gain, bias, x, targets, rng=np.random, E=None):
        self.n_calls[self] += 1
        if E is None:
            return np.random.rand(x.shape[1], targets.shape[1]), {"info": "v"}
        else:
            return np.random.rand(x.shape[1], E.shape[1]), {"info": "v"}


def get_solver_test_args(**kwargs):
    M = 100
    N = 10
    D = 2
    conn = nengo.Connection(
        nengo.Ensemble(N, D, add_to_container=False),
        nengo.Node(size_in=D, add_to_container=False),
        add_to_container=False,
    )
    conn.solver = kwargs.pop("solver", nengo.solvers.LstsqL2nz())
    defaults = {
        "conn": conn,
        "gain": np.ones(N),
        "bias": np.ones(N),
        "x": np.ones((M, D)),
        "targets": np.ones((M, N)),
        "rng": np.random.RandomState(42),
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
        solver=nengo.solvers.LstsqL2nz(),
        add_to_container=False,
    )
    return {
        "conn": conn,
        "gain": np.ones(N),
        "bias": np.ones(N),
        "x": np.ones((M, D)),
        "targets": np.ones((M, N)),
        "rng": np.random.RandomState(42),
        "E": np.ones((D, N2)),
    }


def test_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    # Basic test, that results are cached.
    with DecoderCache(cache_dir=cache_dir) as cache:
        solver_mock = SolverMock()
        decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
            **get_solver_test_args()
        )
        assert SolverMock.n_calls[solver_mock] == 1
        decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
            **get_solver_test_args()
        )
        assert SolverMock.n_calls[solver_mock] == 1  # result read from cache?
        assert_equal(decoders1, decoders2)
        assert solver_info1 == solver_info2

        solver_args = get_solver_test_args()
        solver_args["gain"] *= 2
        decoders3, solver_info3 = cache.wrap_solver(solver_mock)(**solver_args)
        assert SolverMock.n_calls[solver_mock] == 2
        assert np.any(decoders1 != decoders3)

        # Test that the cache does not load results of another solver.
        another_solver = SolverMock()
        cache.wrap_solver(another_solver)(
            **get_solver_test_args(solver=nengo.solvers.LstsqNoise())
        )
        assert SolverMock.n_calls[another_solver] == 1


def test_corrupted_decoder_cache(tmpdir):
    cache_dir = str(tmpdir)

    with DecoderCache(cache_dir=cache_dir) as cache:
        solver_mock = SolverMock()
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 1

        # corrupt the cache
        for path in cache.get_files():
            with open(path, "w") as f:
                f.write("corrupted")

        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 2


def test_corrupted_decoder_cache_index(tmpdir):
    cache_dir = str(tmpdir)

    with DecoderCache(cache_dir=cache_dir):
        pass  # Initialize cache with required files
    assert len(os.listdir(cache_dir)) == 2  # index, index.lock

    # Write corrupted index
    with open(os.path.join(cache_dir, CacheIndex._INDEX), "w") as f:
        f.write("(d")  # empty dict, but missing '.' at the end

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
            "Test succeeded by chance. Adjust get_solver_test_args() to "
            "produce date not aligned with the files system fragment size."
        )

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
        cache.wrap_solver(another_solver)(
            **get_solver_test_args(solver=nengo.solvers.LstsqNoise())
        )

        cache_size = cache.get_size_in_bytes()
        assert cache_size > 0

        cache.shrink(cache_size - 1)

        # check that older cached result was removed
        assert SolverMock.n_calls[solver_mock] == 1
        cache.wrap_solver(another_solver)(
            **get_solver_test_args(solver=nengo.solvers.LstsqNoise())
        )
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

        cache.wrap_solver(solver_mock)(
            **get_solver_test_args(solver=nengo.solvers.LstsqNoise())
        )

        cache_size = cache.get_size_in_bytes()
        assert cache_size > 0

        def raise_file_not_found(orig_fn):
            def fn(filename, *args, **kwargs):
                if filename.endswith(".lock"):
                    return orig_fn(filename, *args, **kwargs)
                raise OSError(errno.ENOENT, "File not found.")

            return fn

        monkeypatch.setattr(cache, "get_size_in_bytes", lambda: cache_size)
        monkeypatch.setattr("os.stat", raise_file_not_found(os.stat))
        monkeypatch.setattr("os.remove", raise_file_not_found(os.remove))
        monkeypatch.setattr("os.unlink", raise_file_not_found(os.unlink))

        cache.shrink(limit)


def test_decoder_cache_with_E_argument_to_solver(tmpdir):
    cache_dir = str(tmpdir)
    solver_mock = SolverMock()

    with DecoderCache(cache_dir=cache_dir) as cache:
        decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
            **get_weight_solver_test_args()
        )
        assert SolverMock.n_calls[solver_mock] == 1
        decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
            **get_weight_solver_test_args()
        )
        assert SolverMock.n_calls[solver_mock] == 1  # read from cache?
        assert_equal(decoders1, decoders2)
        assert solver_info1 == solver_info2


class DummyA:
    def __init__(self, attr=0):
        self.attr = attr


nengo.cache.Fingerprint.whitelist(DummyA)


class DummyB:
    def __init__(self, attr=0):
        self.attr = attr


nengo.cache.Fingerprint.whitelist(DummyB)


def dummy_fn(arg):
    pass


@pytest.mark.parametrize(
    "reference, equal, different",
    (
        (True, True, False),  # bool
        (False, False, True),  # bool
        (1.0, 1.0, 2.0),  # float
        (1, 1, 2),  # int
        (1.0 + 2.0j, 1 + 2j, 2.0 + 1j),  # complex
        (b"a", b"a", b"b"),  # bytes
        ((0, 1), (0, 1), (0, 2)),  # tuple
        ([0, 1], [0, 1], [0, 2]),  # list
        ("a", "a", "b"),  # str
        (np.eye(2), np.eye(2), np.array([[0, 1], [1, 0]])),  # array
        (DummyA(), DummyA(), DummyB()),  # object instance
        (DummyA(1), DummyA(1), DummyA(2)),  # object instance
        (LstsqL2(reg=0.1), LstsqL2(reg=0.1), LstsqL2(reg=0.2)),  # solver
    ),
)
def test_fingerprinting(reference, equal, different):
    assert str(Fingerprint(reference)) == str(Fingerprint(equal))
    assert str(Fingerprint(reference)) != str(Fingerprint(different))


@pytest.mark.parametrize(
    "obj",
    (
        np.array([object()]),  # array
        np.array([(1.0,)], dtype=[("field1", "f8")]),  # array
        {"a": 1, "b": 2},  # dict
        object(),  # object instance
        dummy_fn,  # function
    ),
)
def test_unsupported_fingerprinting(obj):
    with pytest.raises(FingerprintError):
        Fingerprint(obj)


@pytest.mark.parametrize("cls", neuron_types + solver_types + lstsq_solver_types)
def test_supported_fingerprinting(cls, monkeypatch):
    # patch so we can instantiate various solvers without the proper libraries
    monkeypatch.setitem(sys.modules, "scipy", Mock())
    monkeypatch.setitem(sys.modules, "scipy.optimize", Mock())
    monkeypatch.setitem(sys.modules, "sklearn", Mock())
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", Mock())
    monkeypatch.setitem(sys.modules, "sklearn.utils", Mock())
    monkeypatch.setitem(sys.modules, "sklearn.utils.extmath", Mock())

    obj = cls()
    assert Fingerprint.supports(obj)

    # check fingerprint is created without error and is a valid sha1 hash
    fp = str(Fingerprint(obj))
    assert len(fp) == 40 and set(fp).issubset("0123456789abcdef")


def test_fails_for_lambda_expression():
    with pytest.raises(FingerprintError):
        Fingerprint(lambda x: x)


def test_cache_works(tmpdir, Simulator, seed):
    cache_dir = str(tmpdir)

    model = nengo.Network(seed=seed)
    with model:
        nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    assert len(os.listdir(cache_dir)) == 0
    with Simulator(
        model,
        model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir)
        ),
    ):
        assert len(os.listdir(cache_dir)) == 3  # index, index.lock, and *.nco


def test_cache_not_used_without_seed(tmpdir, Simulator):
    cache_dir = str(tmpdir)

    model = nengo.Network()
    with model:
        nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    assert len(os.listdir(cache_dir)) == 0
    with Simulator(
        model,
        model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir)
        ),
    ):
        assert len(os.listdir(cache_dir)) == 2  # index, index.lock


def build_many_ensembles(cache_dir, Simulator):
    with nengo.Network(seed=1) as model:
        for _ in range(100):
            nengo.Connection(nengo.Ensemble(10, 1), nengo.Ensemble(10, 1))

    with Simulator(
        model,
        model=nengo.builder.Model(
            dt=0.001, decoder_cache=DecoderCache(cache_dir=cache_dir)
        ),
    ):
        pass


@pytest.mark.slow
def test_cache_concurrency(tmpdir, Simulator):
    cache_dir = str(tmpdir)

    n_processes = 100
    processes = [
        multiprocessing.Process(
            target=build_many_ensembles, args=(cache_dir, Simulator)
        )
        for _ in range(n_processes)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join(60)
    for p in processes:
        assert p.exitcode == 0


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

    monkeypatch.setattr(os, "replace", raise_error)
    with pytest.warns(CacheIOWarning):
        with WriteableCacheIndex(cache_dir=str(tmpdir)):
            pass


def test_shrink_does_not_fail_if_lock_cannot_be_acquired(tmpdir):
    cache = DecoderCache(cache_dir=str(tmpdir))
    cache._index._lock.timeout = 1.0
    with cache:
        cache.wrap_solver(SolverMock())(**get_solver_test_args())
    with cache._index._lock:
        cache.shrink(limit=0)
