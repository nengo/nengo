"""Caching capabilities for a faster build process."""

import errno
import hashlib
import logging
import os
import pickle
import shutil
import struct
from subprocess import CalledProcessError
import sys
from uuid import uuid1
import warnings

import numpy as np

from nengo.exceptions import (
    CacheIOError,
    CacheIOWarning,
    FingerprintError,
    TimeoutError,
)
from nengo.neurons import (
    AdaptiveLIF,
    AdaptiveLIFRate,
    Direct,
    Izhikevich,
    LIF,
    LIFRate,
    RectifiedLinear,
    Sigmoid,
    SpikingRectifiedLinear,
)
from nengo.rc import rc
from nengo.solvers import (
    Lstsq,
    LstsqDrop,
    LstsqL1,
    LstsqL2,
    LstsqL2nz,
    LstsqMultNoise,
    LstsqNoise,
    Nnls,
    NnlsL2,
    NnlsL2nz,
)
from nengo.utils import nco
from nengo.utils.cache import byte_align, bytes2human, human2bytes
from nengo.utils.least_squares_solvers import (
    Cholesky,
    ConjgradScipy,
    LSMRScipy,
    Conjgrad,
    BlockConjgrad,
    SVD,
    RandomizedSVD,
)
from nengo.utils.lock import FileLock

logger = logging.getLogger(__name__)

if sys.version_info < (3, 3, 0):
    # there was no PermissionError before 3.3
    PermissionError = OSError


def get_fragment_size(path):
    try:
        return os.statvfs(path).f_frsize
    except AttributeError:  # no statvfs on Windows
        return 4096  # correct value in 99% of cases


def safe_stat(path):
    """Does os.stat, but fails gracefully in case of an OSError."""
    try:
        return os.stat(path)
    except OSError as err:
        logger.warning("OSError during safe_stat: %s", err)
    return None


def safe_remove(path):
    """Does os.remove, but fails gracefully in case of an OSError."""
    try:
        os.remove(path)
    except OSError as err:
        logger.warning("OSError during safe_remove: %s", err)


def safe_makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as err:
            logger.warning("OSError during safe_makedirs: %s", err)


def check_dtype(ndarray):
    return ndarray.dtype.isbuiltin == 1 and not ndarray.dtype.hasobject


def check_seq(tpl):
    return all(Fingerprint.supports(x) for x in tpl)


def check_attrs(obj):
    attrs = [getattr(obj, x) for x in dir(obj) if not x.startswith("_")]
    return all(Fingerprint.supports(x) for x in attrs if not callable(x))


class Fingerprint:
    """Fingerprint of an object instance.

    A finger print is equal for two instances if and only if they are of the
    same type and have the same attributes.

    The fingerprint will be used as identification for caching.

    Parameters
    ----------
    obj : object
        Object to fingerprint.

    Attributes
    ----------
    fingerprint : hash
        A unique fingerprint for the object instance.

    Notes
    -----
    Not all objects can be fingerprinted. In particular, custom classes are
    tricky to fingerprint as their implementation can change without changing
    its fingerprint, as the type and attributes may be the same.

    In order to ensure that only safe object are fingerprinted, this class
    maintains class attribute ``WHITELIST`` that contains all types that can
    be safely fingerprinted.

    If you want your custom class to be fingerprinted, call the
    `.whitelist` class method and pass in your class.
    """

    __slots__ = ("fingerprint",)

    SOLVERS = (
        Lstsq,
        LstsqDrop,
        LstsqL1,
        LstsqL2,
        LstsqL2nz,
        LstsqNoise,
        LstsqMultNoise,
        Nnls,
        NnlsL2,
        NnlsL2nz,
    )
    LSTSQ_METHODS = (
        BlockConjgrad,
        Cholesky,
        Conjgrad,
        ConjgradScipy,
        LSMRScipy,
        RandomizedSVD,
        SVD,
    )
    NEURON_TYPES = (
        AdaptiveLIF,
        AdaptiveLIFRate,
        Direct,
        Izhikevich,
        LIF,
        LIFRate,
        RectifiedLinear,
        Sigmoid,
        SpikingRectifiedLinear,
    )

    WHITELIST = set(
        (type(None), bool, float, complex, bytes, list, tuple, np.ndarray, int, str)
        + SOLVERS
        + LSTSQ_METHODS
        + NEURON_TYPES
    )
    CHECKS = dict(
        [(np.ndarray, check_dtype), (tuple, check_seq), (list, check_seq)]
        + [(_x, check_attrs) for _x in SOLVERS + LSTSQ_METHODS + NEURON_TYPES]
    )

    def __init__(self, obj):
        if not self.supports(obj):
            raise FingerprintError(
                "Object of type %r cannot be fingerprinted." % type(obj).__name__
            )

        self.fingerprint = hashlib.sha1()
        try:
            self.fingerprint.update(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        except Exception as err:
            raise FingerprintError(str(err))

    def __str__(self):
        return self.fingerprint.hexdigest()

    @classmethod
    def supports(cls, obj):
        """Determines whether ``obj`` can be fingerprinted.

        Uses the `.whitelist`  method and runs the check function associated
        with the type of ``obj``.
        """
        typ = type(obj)
        in_whitelist = typ in cls.WHITELIST
        succeeded_check = typ not in cls.CHECKS or cls.CHECKS[typ](obj)
        return in_whitelist and succeeded_check

    @classmethod
    def whitelist(cls, typ, fn=None):
        """Whitelist the type given in ``typ``.

        Will run the check function ``fn`` on objects if provided.
        """
        cls.WHITELIST.add(typ)
        if fn is not None:
            cls.CHECKS[typ] = fn


class CacheIndex:
    """Cache index mapping keys to (filename, start, end) tuples.

    Once instantiated the cache index has to be used in a ``with`` block to
    allow access. The index will not be loaded before the ``with`` block is
    entered.

    This class only provides read access to the cache index. For write
    access use `.WriteableCacheIndex`.

    Examples
    --------
    >>> cache_dir = "./my_cache"
    >>> to_cache = ("gotta cache 'em all", 151)
    >>> with CacheIndex(cache_dir) as index:
    ...     filename, start, end = index[hash(to_cache)]

    Parameters
    ----------
    cache_dir : str
        Path where the cache is stored.

    Attributes
    ----------
    cache_dir : str
        Path where the cache is stored.
    index_path : str
        Path to the cache index file.
    legacy_path : str
        Path to a potentially existing legacy file. This file was previously
        used to track version information, but is now obsolete. It will still
        be used to retrieve version information in case of an old cache index
        that does not store version information itself.
    version : tuple
        Version code of the loaded cache index. The first element gives the
        format of the cache and the second element gives the pickle protocol
        used to store the index. Note that a cache index will always be written
        in the newest version with the highest pickle protocol.
    VERSION (class attribute) : int
        Highest supported version, and version used to store the cache index.

    Notes
    -----
    Under the hood, the cache index is stored as a pickle file. The pickle file
    contains two objects which are read sequentially: the ``version`` tuple,
    and the ``index`` dictionary mapping keys to (filename, start, end) tuples.
    Previously, these two objects were stored separately, with the ``version``
    tuple stored in a ``legacy.txt`` file. These two objects were consolidated
    in the pickle file in Nengo 2.3.0.
    """

    _INDEX = "index"
    _LEGACY = "legacy.txt"
    VERSION = 2

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.version = None
        self._index = None

    @property
    def index_path(self):
        return os.path.join(self.cache_dir, self._INDEX)

    @property
    def legacy_path(self):
        return os.path.join(self.cache_dir, self._LEGACY)

    def __contains__(self, key):
        return key in self._index

    def __getitem__(self, key):
        return self._index[key]

    def __setitem__(self, key):
        raise TypeError("Index is readonly.")

    def __delitem__(self, key):
        raise TypeError("Index is readonly.")

    def __enter__(self):
        self._load_index_file()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _load_index_file(self):
        with open(self.index_path, "rb") as f:
            self.version = pickle.load(f)
            if isinstance(self.version, tuple):
                if (
                    self.version[0] > self.VERSION
                    or self.version[1] > pickle.HIGHEST_PROTOCOL
                ):
                    raise CacheIOError("Unsupported cache index file format.")
                self._index = pickle.load(f)
            else:
                self._index = self.version
                self.version = self._get_legacy_version()
        assert isinstance(self.version, tuple)
        assert isinstance(self._index, dict)

    def _get_legacy_version(self):
        try:
            with open(self.legacy_path, "r") as lf:
                text = lf.read()
            return tuple(int(x.strip()) for x in text.split("."))
        except Exception:
            logger.exception("Decoder cache version information could not be read.")
            return (-1, -1)


class WriteableCacheIndex(CacheIndex):
    """Writable cache index mapping keys to files.

    This class allows write access to the cache index.

    The updated cache file will be written when the ``with`` block is exited.
    The initial read and the write on exit of the ``with`` block are locked
    against concurrent access with a file lock. The lock will be released
    within the ``with`` block.

    Examples
    --------
    >>> cache_dir = "./my_cache"
    >>> to_cache = ("gotta cache 'em all", 151)
    >>> key = hash(to_cache)
    >>> with WriteableCacheIndex(cache_dir) as index:
    ...     index[key] = ("file1", 0, 1)  # set an item
    ...     del index[key]  # remove an item by key
    ...     index.remove_file_entry("file1")  # remove an item by filename

    Parameters
    ----------
    cache_dir : str
        Path where the cache is stored.
    """

    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self._lock = FileLock(self.index_path + ".lock")
        self._updates = {}
        self._deletes = set()
        self._removed_files = set()

    def __getitem__(self, key):
        if key in self._updates:
            return self._updates[key]
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("Cache entries must include filename, start, and end.")
        self._updates[key] = value

    def __delitem__(self, key):
        self._deletes.add(key)

    def __enter__(self):
        with self._lock:
            try:
                self._load_index_file()
            except Exception:
                logger.exception("Decoder cache index corrupted. Reinitializing cache.")
                # If we can't load the index file, the cache is corrupted,
                # so we invalidate it (delete all files in the cache)
                self._reinit()
                self._write_index()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sync()

    def _reinit(self):
        for f in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, f)
            if path == self._lock.filename:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        self._index = {}

    def remove_file_entry(self, filename):
        """Remove entries mapping to ``filename``."""
        if os.path.realpath(filename).startswith(self.cache_dir):
            filename = os.path.relpath(filename, self.cache_dir)
        self._removed_files.add(filename)

    def _write_index(self):
        assert self._lock.acquired
        with open(self.index_path + ".part", "wb") as f:
            # Use protocol 2 for version information to ensure that
            # all Python versions supported by Nengo will be able to
            # read it in the future.
            pickle.dump((self.VERSION, pickle.HIGHEST_PROTOCOL), f, 2)
            # Use highest available protocol for index data for maximum
            # performance.
            pickle.dump(self._index, f, pickle.HIGHEST_PROTOCOL)
        try:
            os.replace(self.index_path + ".part", self.index_path)
        except (CalledProcessError, PermissionError):
            # It may fail when
            # another program like a virus scanner is accessing the file to be
            # moved. There is not a lot we could do about this. See
            # <https://github.com/nengo/nengo/issues/1200> for more info.
            warnings.warn(
                "The cache index could not be updated because another program "
                "blocked access to it. This is commonly caused by anti-virus "
                "software. It is safe to ignore this warning. But if you see "
                "it a lot, you might want to consider doing one of the "
                "following for the best Nengo performance:\n"
                "1. Configure your anti-virus to ignore the Nengo cache "
                "folder ('{cache_dir}').\n"
                "2. Disable the cache.\n".format(cache_dir=self.cache_dir),
                category=CacheIOWarning,
            )

        if os.path.exists(self.legacy_path):
            os.remove(self.legacy_path)

    def sync(self):
        """Write changes to the cache index back to disk.

        The call to this function will be locked by a file lock.
        """
        try:
            with self._lock:
                try:
                    self._load_index_file()
                except IOError as err:
                    if err.errno == errno.ENOENT:
                        self._index = {}
                    else:
                        raise

                self._index.update(self._updates)
                for key in self._deletes:
                    del self._index[key]
                self._index = {
                    k: v
                    for k, v in self._index.items()
                    if v[0] not in self._removed_files
                }

                self._write_index()
        except TimeoutError:
            warnings.warn(
                "Decoder cache index could not acquire lock. "
                "Cache index was not synced."
            )

        self._updates.clear()
        self._deletes.clear()
        self._removed_files.clear()


class DecoderCache:
    """Cache for decoders.

    Hashes the arguments to the decoder solver and stores the result in a file
    which will be reused in later calls with the same arguments.

    Be aware that decoders should not use any global state, but only values
    passed and attributes of the object instance. Otherwise the wrong solver
    results might get loaded from the cache.

    Parameters
    ----------
    readonly : bool
        Indicates that already existing items in the cache will be used, but no
        new items will be written to the disk in case of a cache miss.
    cache_dir : str or None
        Path to the directory in which the cache will be stored. It will be
        created if it does not exists. Will use the value returned by
        `.get_default_dir`, if ``None``.
    """

    _CACHE_EXT = ".nco"

    def __init__(self, readonly=False, cache_dir=None):
        self.readonly = readonly
        if cache_dir is None:
            cache_dir = self.get_default_dir()
        self.cache_dir = cache_dir
        if readonly:
            self._index = CacheIndex(cache_dir)
        else:
            safe_makedirs(self.cache_dir)
            self._index = WriteableCacheIndex(cache_dir)
        self._fragment_size = get_fragment_size(self.cache_dir)
        self._fd = None
        self._in_context = False

    def __enter__(self):
        try:
            try:
                self._index.__enter__()
            except TimeoutError:
                self.readonly = True
                self._index = CacheIndex(self.cache_dir)
                self._index.__enter__()
                warnings.warn(
                    "Decoder cache could not acquire lock and was "
                    "set to readonly mode."
                )
        except Exception as e:
            self.readonly = True
            self._index = None
            logger.debug("Could not acquire lock because: %s", e)
            warnings.warn("Decoder cache could not acquire lock and was deactivated.")
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._in_context = False
        self._close_fd()
        if self._index is not None:
            rval = self._index.__exit__(exc_type, exc_value, traceback)
            return rval

    @staticmethod
    def get_default_dir():
        """Returns the default location of the cache.

        Returns
        -------
        str
        """
        return rc.get("decoder_cache", "path")

    def _close_fd(self):
        if self._fd is not None:
            self._fd.close()
            self._fd = None

    def _get_fd(self):
        if self._fd is None:
            self._fd = open(self._key2path(str(uuid1())), "wb")
        return self._fd

    def get_files(self):
        """Returns all of the files in the cache.

        Returns
        -------
        list of (str, int) tuples
        """
        files = []
        for subdir in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, subdir)
            if os.path.isdir(path):
                files.extend(os.path.join(path, f) for f in os.listdir(path))
        return files

    def get_size(self):
        """Returns the size of the cache with units as a string.

        Returns
        -------
        str
        """
        return bytes2human(self.get_size_in_bytes())

    def get_size_in_bytes(self):
        """Returns the size of the cache in bytes as an int.

        Returns
        -------
        int
        """
        stats = (safe_stat(f) for f in self.get_files())
        return sum(
            byte_align(st.st_size, self._fragment_size)
            for st in stats
            if st is not None
        )

    def invalidate(self):
        """Invalidates the cache (i.e. removes all cache files)."""
        if self.readonly:
            raise CacheIOError("Cannot invalidate a readonly cache.")
        self._close_fd()
        with self._index:
            for path in self.get_files():
                self.remove_file(path)

    def shrink(self, limit=None):  # noqa: C901
        """Reduces the size of the cache to meet a limit.

        Parameters
        ----------
        limit : int, optional
            Maximum size of the cache in bytes.
        """
        if self.readonly:
            logger.info("Tried to shrink a readonly cache.")
            return

        if limit is None:
            limit = rc.get("decoder_cache", "size")
        if isinstance(limit, str):
            limit = human2bytes(limit)

        self._close_fd()

        fileinfo = []
        excess = -limit
        for path in self.get_files():
            stat = safe_stat(path)
            if stat is not None:
                aligned_size = byte_align(stat.st_size, self._fragment_size)
                excess += aligned_size
                fileinfo.append((stat.st_atime, aligned_size, path))

        # Remove the least recently accessed first
        fileinfo.sort()

        try:
            with self._index:
                for _, size, path in fileinfo:
                    if excess <= 0:
                        break

                    excess -= size
                    self.remove_file(path)
        except TimeoutError:
            logger.debug("Not shrinking cache. Lock could not be acquired.")

    def remove_file(self, path):
        """Removes the file at ``path`` from the cache."""
        self._index.remove_file_entry(path)
        safe_remove(path)

    def wrap_solver(self, solver_fn):  # noqa: C901
        """Takes a decoder solver and wraps it to use caching.

        Parameters
        ----------
        solver : func
            Decoder solver to wrap for caching.

        Returns
        -------
        func
            Wrapped decoder solver.
        """

        def cached_solver(
            conn, gain, bias, x, targets, rng=np.random, **uncached_kwargs
        ):
            if not self._in_context:
                warnings.warn(
                    "Cannot use cached solver outside of " "`with cache` block."
                )
                return solver_fn(
                    conn, gain, bias, x, targets, rng=rng, **uncached_kwargs
                )

            try:
                key = self._get_cache_key(
                    conn.solver, conn.pre_obj.neuron_type, gain, bias, x, targets, rng
                )
            except FingerprintError as e:
                logger.debug("Failed to generate cache key: %s", e)
                return solver_fn(
                    conn, gain, bias, x, targets, rng=rng, **uncached_kwargs
                )

            try:
                path, start, end = self._index[key]
                if self._fd is not None:
                    self._fd.flush()
                with open(path, "rb") as f:
                    f.seek(start)
                    info, decoders = nco.read(f)
            except Exception as err:
                if isinstance(err, KeyError):
                    logger.debug("Cache miss [%s].", key)
                else:
                    logger.exception("Corrupted cache entry [%s].", key)
                decoders, info = solver_fn(
                    conn, gain, bias, x, targets, rng=rng, **uncached_kwargs
                )
                if not self.readonly:
                    fd = self._get_fd()
                    start = fd.tell()
                    nco.write(fd, info, decoders)
                    end = fd.tell()
                    self._index[key] = (fd.name, start, end)
            else:
                logger.debug("Cache hit [%s]: Loaded stored decoders.", key)
            return decoders, info

        return cached_solver

    def _get_cache_key(self, solver, neuron_type, gain, bias, x, targets, rng):
        h = hashlib.sha1()

        h.update(str(Fingerprint(solver)).encode("utf-8"))
        h.update(str(Fingerprint(neuron_type)).encode("utf-8"))

        h.update(np.ascontiguousarray(gain).data)
        h.update(np.ascontiguousarray(bias).data)
        h.update(np.ascontiguousarray(x).data)
        h.update(np.ascontiguousarray(targets).data)

        # rng format doc:
        # https://docs.scipy.org/doc/numpy/reference/random/generated/numpy.random.mtrand.RandomState.get_state.html#numpy.random.mtrand.RandomState.get_state
        state = rng.get_state()
        h.update(state[0].encode())  # string 'MT19937'
        h.update(state[1].data)  # 1-D array of 624 unsigned integer keys
        h.update(struct.pack("q", state[2]))  # integer pos
        h.update(struct.pack("q", state[3]))  # integer has_gauss
        h.update(struct.pack("d", state[4]))  # float cached_gaussian

        return h.hexdigest()

    def _key2path(self, key):
        prefix = key[:2]
        suffix = key[2:]
        directory = os.path.join(self.cache_dir, prefix)
        safe_makedirs(directory)
        return os.path.join(directory, suffix + self._CACHE_EXT)


class NoDecoderCache:
    """Provides the same interface as `.DecoderCache` without caching."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def wrap_solver(self, solver_fn):
        return solver_fn

    def get_size_in_bytes(self):
        return 0

    def get_size(self):
        return "0 B"

    def shrink(self, limit=0):
        pass

    def invalidate(self):
        pass


def get_default_decoder_cache():
    if rc.getboolean("decoder_cache", "enabled"):
        decoder_cache = DecoderCache(rc.getboolean("decoder_cache", "readonly"))
    else:
        decoder_cache = NoDecoderCache()
    return decoder_cache
