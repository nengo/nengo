import errno
import logging
import multiprocessing
import os
import pickle
import sys
from pathlib import Path
from subprocess import CalledProcessError

import numpy as np
import pytest
from numpy.testing import assert_equal

import nengo
import nengo.neurons
import nengo.utils.least_squares_solvers
from nengo.cache import (
    CacheIndex,
    CacheIOError,
    DecoderCache,
    Fingerprint,
    NoDecoderCache,
    WriteableCacheIndex,
    get_fragment_size,
    safe_stat,
)
from nengo.exceptions import CacheIOWarning, FingerprintError
from nengo.solvers import LstsqL2


class Mock:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    def __getitem__(self, key):
        return Mock()

    def __iter__(self):
        return iter([])

    def __mul__(self, other):
        return 1.0

    @classmethod
    def __getattr__(cls, name):
        if name in ("__file__", "__path__"):
            return "/dev/null"
        if name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        return Mock()


def isstrictsubclass(x, cls):
    return isinstance(x, type) and issubclass(x, cls) and x is not cls


def list_objects(module):
    return [getattr(module, key) for key in dir(module) if not key.startswith("_")]


neuron_types = [
    obj
    for obj in list_objects(nengo.neurons)
    if isstrictsubclass(obj, nengo.neurons.NeuronType)
    and obj is not nengo.neurons.RatesToSpikesNeuronType
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
        return np.random.rand(x.shape[1], E.shape[1]), {"info": "v"}


def get_solver_test_args(
    n_eval_points=100,
    n_neurons=10,
    dimensions=2,
    solver=nengo.solvers.LstsqL2nz(),
    weights=False,
):
    conn = nengo.Connection(
        nengo.Ensemble(n_neurons, dimensions, add_to_container=False),
        nengo.Node(size_in=dimensions, add_to_container=False),
        solver=solver,
        add_to_container=False,
    )
    defaults = {
        "conn": conn,
        "gain": np.ones(n_neurons),
        "bias": np.ones(n_neurons),
        "x": np.ones((n_eval_points, dimensions)),
        "targets": np.ones((n_eval_points, n_neurons)),
        "rng": np.random.RandomState(42),
    }
    if weights:
        defaults["E"] = np.ones((dimensions, n_neurons // 2))
    return defaults


def test_decoder_cache(tmp_path):
    cache_dir = str(tmp_path)

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
        decoders3, _ = cache.wrap_solver(solver_mock)(**solver_args)
        assert SolverMock.n_calls[solver_mock] == 2
        assert np.any(decoders1 != decoders3)

        # Test that the cache does not load results of another solver.
        another_solver = SolverMock()
        cache.wrap_solver(another_solver)(
            **get_solver_test_args(solver=nengo.solvers.LstsqNoise())
        )
        assert SolverMock.n_calls[another_solver] == 1

    with DecoderCache(cache_dir=cache_dir, readonly=True) as cache:
        n_calls = int(SolverMock.n_calls[solver_mock])
        decoders3, solver_info3 = cache.wrap_solver(solver_mock)(
            **get_solver_test_args()
        )
        assert SolverMock.n_calls[solver_mock] == n_calls
        assert_equal(decoders3, decoders2)
        assert solver_info3 == solver_info2


def test_corrupted_decoder_cache(tmp_path):
    cache_dir = str(tmp_path)

    with DecoderCache(cache_dir=cache_dir) as cache:
        solver_mock = SolverMock()
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 1

        # corrupt the cache
        for path in cache.get_files():
            with open(path, "w", encoding="utf-8") as f:
                f.write("corrupted")

        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 2


def test_corrupted_decoder_cache_index(tmp_path):
    cache_dir = str(tmp_path)

    with DecoderCache(cache_dir=cache_dir):
        pass  # Initialize cache with required files
    assert len(os.listdir(cache_dir)) == 2  # index, index.lock

    # Write corrupted index
    with open(os.path.join(cache_dir, CacheIndex._INDEX), "w", encoding="utf-8") as f:
        f.write("(d")  # empty dict, but missing '.' at the end

    # Try to load index
    with DecoderCache(cache_dir=cache_dir):
        pass
    assert len(os.listdir(cache_dir)) == 2  # index, index.lock


def test_too_new_decoder_cache_index(tmp_path):
    cache_dir = str(tmp_path)

    # Write index with super large version numbers
    with open(os.path.join(cache_dir, CacheIndex._INDEX), "wb") as f:
        pickle.dump((1000, 1000), f)

    with pytest.warns(UserWarning, match="could not acquire lock and was deactivated"):
        with DecoderCache(cache_dir=cache_dir) as cache:
            solver_mock = SolverMock()
            cache.wrap_solver(solver_mock)(**get_solver_test_args())
            assert SolverMock.n_calls[solver_mock] == 1


def test_decoder_cache_invalidation(tmp_path):
    cache_dir = str(tmp_path)
    solver_mock = SolverMock()

    # Basic test, that results are cached.
    with DecoderCache(cache_dir=cache_dir) as cache:
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 1
        cache.invalidate()
        cache.wrap_solver(solver_mock)(**get_solver_test_args())
        assert SolverMock.n_calls[solver_mock] == 2


def test_readonly_cache(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    cache_dir = str(tmp_path)

    with open(os.path.join(cache_dir, CacheIndex._INDEX), "wb") as f:
        pickle.dump((CacheIndex.VERSION, pickle.HIGHEST_PROTOCOL), f)
        pickle.dump({}, f)

    with DecoderCache(readonly=True, cache_dir=cache_dir) as cache:
        cache.shrink()
        with pytest.raises(CacheIOError, match="Cannot invalidate a readonly cache."):
            cache.invalidate()
    assert len(caplog.records) == 1
    assert caplog.records[0].message == "Tried to shrink a readonly cache."


def test_decoder_cache_size_includes_overhead(tmp_path):
    cache_dir = str(tmp_path)
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


def test_decoder_cache_shrinking(tmp_path):
    cache_dir = str(tmp_path)
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


def test_decoder_cache_shrink_threadsafe(monkeypatch, tmp_path):
    """Tests that shrink handles files deleted by other processes."""
    cache_dir = str(tmp_path)
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
                if filename.suffix == ".lock":
                    return orig_fn(filename, *args, **kwargs)
                raise OSError(errno.ENOENT, "File not found.")

            return fn

        monkeypatch.setattr(cache, "get_size_in_bytes", lambda: cache_size)
        monkeypatch.setattr("os.stat", raise_file_not_found(os.stat))
        monkeypatch.setattr("os.remove", raise_file_not_found(os.remove))
        monkeypatch.setattr("os.unlink", raise_file_not_found(os.unlink))

        cache.shrink(limit)


def test_decoder_cache_with_E_argument_to_solver(tmp_path):
    cache_dir = str(tmp_path)
    solver_mock = SolverMock()

    with DecoderCache(cache_dir=cache_dir) as cache:
        decoders1, solver_info1 = cache.wrap_solver(solver_mock)(
            **get_solver_test_args(weights=True)
        )
        assert SolverMock.n_calls[solver_mock] == 1
        decoders2, solver_info2 = cache.wrap_solver(solver_mock)(
            **get_solver_test_args(weights=True)
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


nengo.cache.Fingerprint.whitelist(DummyB, fn=lambda obj: True)


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
        object(),  # object instance
        dummy_fn,  # function
    ),
)
def test_unsupported_fingerprinting(obj):
    with pytest.raises(FingerprintError, match="cannot be fingerprinted"):
        Fingerprint(obj)


@pytest.mark.parametrize("cls", neuron_types + solver_types + lstsq_solver_types)
def test_supported_fingerprinting(cls, monkeypatch):
    # patch so we can instantiate various solvers without the proper libraries
    monkeypatch.setitem(sys.modules, "scipy", Mock())
    monkeypatch.setitem(sys.modules, "scipy.optimize", Mock())
    monkeypatch.setitem(sys.modules, "scipy.sparse", Mock())
    monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", Mock())
    monkeypatch.setitem(sys.modules, "sklearn", Mock())
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", Mock())
    monkeypatch.setitem(sys.modules, "sklearn.utils", Mock())
    monkeypatch.setitem(sys.modules, "sklearn.utils.extmath", Mock())

    args = []
    if issubclass(cls, nengo.neurons.RatesToSpikesNeuronType):
        # spiking types require a `base_type` argument, so provide one
        args.append(nengo.neurons.RectifiedLinear())

    obj = cls(*args)
    assert Fingerprint.supports(obj)

    # check fingerprint is created without error and is a valid sha1 hash
    fp = str(Fingerprint(obj))
    assert len(fp) == 40 and set(fp).issubset("0123456789abcdef")


def test_fails_for_lambda_expression():
    with pytest.raises(FingerprintError, match="cannot be fingerprinted"):
        Fingerprint(lambda x: x)


def test_cache_works(tmp_path, Simulator, seed):
    cache_dir = str(tmp_path)

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


def test_cache_not_used_without_seed(tmp_path, Simulator):
    cache_dir = str(tmp_path)

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
def test_cache_concurrency(tmp_path, Simulator):
    cache_dir = str(tmp_path)

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


def test_warns_out_of_context(tmp_path):
    cache_dir = str(tmp_path)
    cache = DecoderCache(cache_dir=cache_dir)

    solver_mock = SolverMock()
    solver = cache.wrap_solver(solver_mock)
    with pytest.warns(UserWarning):
        solver(**get_solver_test_args())
    assert SolverMock.n_calls[solver_mock] == 1


def test_cacheindex_cannot_write(tmp_path):
    index = WriteableCacheIndex(cache_dir=str(tmp_path))
    with index:
        index[0] = ("file0", 0, 0)
    mtime = os.stat(index.index_path).st_mtime

    index = CacheIndex(cache_dir=str(tmp_path))
    with index:
        with pytest.raises(TypeError):
            index[0] = ("file", 0, 0)
        with pytest.raises(TypeError):
            del index[0]
        assert index[0] == ("file0", 0, 0)
    assert os.stat(index.index_path).st_mtime == mtime


def test_writeablecacheindex_writes(tmp_path):
    index = WriteableCacheIndex(cache_dir=str(tmp_path))
    with index:
        index[0] = ("file0", 0, 0)
        index[1] = ("file1", 0, 0)
        del index[1]

    # Verify with readonly cacheindex
    index = CacheIndex(cache_dir=str(tmp_path))
    with index:
        assert index[0] == ("file0", 0, 0)
        assert 1 not in index


def test_writeablecacheindex_setitem(tmp_path):
    index = WriteableCacheIndex(cache_dir=str(tmp_path))

    with pytest.raises(ValueError):
        index[0] = "file0"
    with pytest.raises(ValueError):
        index[0] = ("file0", 0)
    with pytest.raises(ValueError):
        index[0] = ("file0", 0, 0, 0)


def test_writeablecacheindex_removes(tmp_path):
    index = WriteableCacheIndex(cache_dir=str(tmp_path))
    with index:
        index[0] = ("file0", 0, 0)
        index[1] = ("file1", 0, 0)
        index.remove_file_entry("file0")
        index.remove_file_entry(os.path.join(str(tmp_path), "file1"))

    # Verify with readonly cacheindex
    index = CacheIndex(cache_dir=str(tmp_path))
    with index:
        assert 0 not in index, "Fails on relative paths"
        assert 1 not in index, "Fails on absolute paths"


def test_writeablecacheindex_warning(monkeypatch, tmp_path):
    with pytest.warns(CacheIOWarning):
        with WriteableCacheIndex(cache_dir=str(tmp_path)) as idx:

            class RaiseError(type(idx.cache_dir)):
                def replace(self, *args, **kwargs):
                    raise CalledProcessError(-1, "move")

            monkeypatch.setattr(idx, "cache_dir", RaiseError(idx.cache_dir))


def test_writeablecacheindex_reinit(tmp_path):
    with WriteableCacheIndex(cache_dir=str(tmp_path)) as idx:
        subdir = idx.cache_dir / "subdir"
        subdir.mkdir()
        file1 = idx.cache_dir / "file1"
        file2 = subdir / "file2"
        with file1.open("w", encoding="utf-8") as fh:
            fh.write("contents1")
        with file2.open("w", encoding="utf-8") as fh:
            fh.write("contents2")

        assert subdir.exists() and file1.exists() and file2.exists()

        idx._reinit()
        assert not (subdir.exists() or file1.exists() or file2.exists())


def test_shrink_does_not_fail_if_lock_cannot_be_acquired(tmp_path):
    cache = DecoderCache(cache_dir=str(tmp_path))
    cache._index._lock.timeout = 1.0
    with cache:
        cache.wrap_solver(SolverMock())(**get_solver_test_args())
    with cache._index._lock:
        cache.shrink(limit=0)


def test_safe_stat(caplog):
    assert safe_stat(Path("/tmp/does/not/exist")) is None
    assert len(caplog.records) == 1
    assert "OSError during safe_stat:" in caplog.records[0].message


def test_no_decoder_cache():
    cache = NoDecoderCache()
    assert cache.get_size_in_bytes() == 0
    assert cache.get_size() == "0 B"
