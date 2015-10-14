"""Caching capabilities for a faster build process."""

from collections import namedtuple
import hashlib
import heapq
import inspect
import logging
import os
import struct
import time

import numpy as np

from nengo.rc import rc
from nengo.utils.cache import byte_align, bytes2human, human2bytes
from nengo.utils.compat import is_string, pickle, PY2
from nengo.utils.lock import FileLock
from nengo.utils import nco

logger = logging.getLogger(__name__)


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


class Fingerprint(object):
    """Fingerprint of an object instance.

    A finger print is equal for two instances if and only if they are of the
    same type and have the same attributes.

    The fingerprint will be used as identification for caching.

    Parameters
    ----------
    obj : object
        Object to fingerprint.
    """

    __slots__ = ['fingerprint']

    def __init__(self, obj):
        self.fingerprint = hashlib.sha1()
        try:
            self.fingerprint.update(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        except (pickle.PicklingError, TypeError) as err:
            raise ValueError("Cannot create fingerprint: {msg}".format(
                msg=str(err)))

    def __str__(self):
        return self.fingerprint.hexdigest()


IndexEntry = namedtuple('IndexEntry', ['atime', 'size', 'key'])


class DecoderCache(object):
    """Cache for decoders.

    Hashes the arguments to the decoder solver and stores the result in a file
    which will be reused in later calls with the same arguments.

    Be aware that decoders should not use any global state, but only values
    passed and attributes of the object instance. Otherwise the wrong solver
    results might get loaded from the cache.

    Parameters
    ----------
    read_only : bool
        Indicates that already existing items in the cache will be used, but no
        new items will be written to the disk in case of a cache miss.
    cache_dir : str or None
        Path to the directory in which the cache will be stored. It will be
        created if it does not exists. Will use the value returned by
        :func:`get_default_dir`, if `None`.
    """

    _CACHE_EXT = '.nco'
    _INDEX = 'index'
    _INDEX_LOCK = 'index.lock'
    _INDEX_STALE = 'index.stale'
    _LEGACY = 'legacy.txt'
    _LEGACY_VERSION = 1
    RESERVED_FILES = ['index', 'index.stale', 'index.lock', 'legacy.txt']

    def __init__(self, read_only=False, cache_dir=None):
        self.read_only = read_only
        if cache_dir is None:
            cache_dir = self.get_default_dir()
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self._fragment_size = get_fragment_size(self.cache_dir)

        self._index_updates = {}
        self._index_lock = FileLock(self._index_lock_file)

        self._update_cache_version()

    def __del__(self):
        if self._index_stale:
            return

        # Write index updates back to disk when object gets destroyed.
        try:
            with self._index_lock:
                with open(self._index_file, 'rb') as f:
                    index = pickle.load(f)

                for k, v in self._index_updates.items():
                    if k not in index or v.atime > index[k].atime:
                        index[k] = v

                with open(self._index_file, 'wb') as f:
                    pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

                self._index_updates = {}
        except:
            self._mark_index_stale()
            raise

    @property
    def _index_lock_file(self):
        return os.path.join(self.cache_dir, self._INDEX_LOCK)

    @property
    def _index_stale_file(self):
        return os.path.join(self.cache_dir, self._INDEX_STALE)

    @property
    def _index_stale(self):
        return os.path.exists(self._index_stale_file)

    def _mark_index_stale(self):
        with open(self._index_stale_file, 'w'):
            pass

    def build_index(self):
        index = {}
        for path in self.get_files():
            stat = safe_stat(path)
            if stat is not None:
                key = self._path2key(path)
                aligned_size = byte_align(stat.st_size, self._fragment_size)
                index[key] = IndexEntry(
                    key=key, atime=stat.st_atime, size=aligned_size)
        return index

    def get_files(self):
        """Returns all of the files in the cache.

        Returns
        -------
        generator of (str, int) tuples
        """
        for subdir in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, subdir)
            if os.path.isdir(path):
                for f in os.listdir(path):
                    yield os.path.join(path, f)

    def get_size_in_bytes(self):
        """Returns the size of the cache in bytes as an int.

        Returns
        -------
        int
        """
        stats = (safe_stat(f) for f in self.get_files())
        return sum(byte_align(st.st_size, self._fragment_size)
                   for st in stats if st is not None)

    def get_size(self):
        """Returns the size of the cache with units as a string.

        Returns
        -------
        str
        """
        return bytes2human(self.get_size_in_bytes())

    def shrink(self, limit=None):
        """Reduces the size of the cache to meet a limit.

        Parameters
        ----------
        limit : int, optional
            Maximum size of the cache in bytes.
        """
        if limit is None:
            limit = rc.get('decoder_cache', 'size')
        if is_string(limit):
            limit = human2bytes(limit)

        with self._index_lock:
            if self._index_stale:
                index = self.build_index()
            else:
                with open(self._index_file, 'rb') as f:
                    index = pickle.load(f)

                for k, v in self._index_updates.items():
                    if k not in index or v.atime > index[k].atime:
                        index[k] = v
            self._index_updates = {}

            excess = sum(x.size for x in index.values()) - limit
            if excess > 0:
                heap = list(index.values())
                heapq.heapify(heap)
                while excess > 0:
                    item = heapq.heappop(heap)
                    excess -= item.size
                    del index[item.key]
                    safe_remove(self._key2path(item.key))

            with open(self._index_file, 'wb') as f:
                pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)

    def invalidate(self):
        """Invalidates the cache (i.e. removes all cache files)."""
        for path in self.get_files():
            safe_remove(path)

    @property
    def _legacy_file(self):
        return os.path.join(self.cache_dir, self._LEGACY)

    def _get_legacy_version(self):
        if os.path.exists(self._legacy_file):
            with open(self._legacy_file, 'r') as lf:
                version = int(lf.read().strip())
        else:
            version = -1
        return version

    def _update_cache_version(self):
        version = self._get_legacy_version()
        if version < 0:
            self._remove_legacy_files()
        if version < 1:
            self._mark_index_stale()
        self._write_legacy_file()

    def _check_legacy_file(self):
        """Checks if the legacy file is up to date."""
        if os.path.exists(self._legacy_file):
            with open(self._legacy_file, 'r') as lf:
                version = int(lf.read().strip())
        else:
            version = -1
        return version == self._LEGACY_VERSION

    def _write_legacy_file(self):
        """Writes a legacy file, indicating that legacy files do not exist."""
        with open(self._legacy_file, 'w') as lf:
            lf.write("%d\n" % self._LEGACY_VERSION)

    def _remove_legacy_files(self):
        """Remove files from now invalid locations in the cache."""
        for f in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, f)
            if (not os.path.isdir(path) and
                    os.path.basename(path) not in self.RESERVED_FILES):
                safe_remove(path)

        self._write_legacy_file()

    @staticmethod
    def get_default_dir():
        """Returns the default location of the cache.

        Returns
        -------
        str
        """
        return rc.get('decoder_cache', 'path')

    def wrap_solver(self, solver_fn):
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
        def cached_solver(solver, neuron_type, gain, bias, x, targets,
                          rng=None, E=None):
            try:
                args, _, _, defaults = inspect.getargspec(solver)
            except TypeError:
                args, _, _, defaults = inspect.getargspec(solver.__call__)
            args = args[-len(defaults):]
            if rng is None and 'rng' in args:
                rng = defaults[args.index('rng')]
            if E is None and 'E' in args:
                E = defaults[args.index('E')]

            key = self._get_cache_key(
                solver_fn, solver, neuron_type, gain, bias, x, targets, rng, E)
            path = self._key2path(key)
            try:
                with open(path, 'rb') as f:
                    solver_info, decoders = nco.read(f)
            except:
                logger.info("Cache miss [{0}].".format(key))
                decoders, solver_info = solver_fn(
                    solver, neuron_type, gain, bias, x, targets, rng=rng, E=E)
                if not self.read_only:
                    with open(path, 'wb') as f:
                        nco.write(f, solver_info, decoders)
            else:
                logger.info(
                    "Cache hit [{0}]: Loaded stored decoders.".format(key))
            self._index_updates[key] = IndexEntry(
                atime=time.time(),
                size=byte_align(decoders.nbytes, self._fragment_size), key=key)
            return decoders, solver_info
        return cached_solver

    @property
    def _index_file(self):
        return os.path.join(self.cache_dir, self._INDEX)

    def _get_cache_key(self, solver_fn, solver, neuron_type, gain, bias,
                       x, targets, rng, E):
        h = hashlib.sha1()

        if PY2:
            h.update(str(Fingerprint(solver_fn)))
            h.update(str(Fingerprint(solver)))
            h.update(str(Fingerprint(neuron_type)))
        else:
            h.update(str(Fingerprint(solver_fn)).encode('utf-8'))
            h.update(str(Fingerprint(solver)).encode('utf-8'))
            h.update(str(Fingerprint(neuron_type)).encode('utf-8'))

        h.update(np.ascontiguousarray(gain).data)
        h.update(np.ascontiguousarray(bias).data)
        h.update(np.ascontiguousarray(x).data)
        h.update(np.ascontiguousarray(targets).data)

        # rng format doc:
        # noqa <http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.get_state.html#numpy.random.RandomState.get_state>
        state = rng.get_state()
        h.update(state[0].encode())  # string 'MT19937'
        h.update(state[1].data)  # 1-D array of 624 unsigned integer keys
        h.update(struct.pack('q', state[2]))  # integer pos
        h.update(struct.pack('q', state[3]))  # integer has_gauss
        h.update(struct.pack('d', state[4]))  # float cached_gaussian

        if E is not None:
            h.update(np.ascontiguousarray(E).data)
        return h.hexdigest()

    def _key2path(self, key):
        prefix = key[:2]
        suffix = key[2:]
        directory = os.path.join(self.cache_dir, prefix)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.join(directory, suffix + self._CACHE_EXT)

    def _path2key(self, path):
        dirname, filename = os.path.split(path)
        prefix = os.path.basename(dirname)
        suffix, _ = os.path.splitext(filename)
        return prefix + suffix


class NoDecoderCache(object):
    """Provides the same interface as :class:`DecoderCache` without caching."""

    def wrap_solver(self, solver_fn):
        return solver_fn

    def get_size_in_bytes(self):
        return 0

    def get_size(self):
        return '0 B'

    def shrink(self, limit=0):
        pass

    def invalidate(self):
        pass


def get_default_decoder_cache():
    if rc.getboolean('decoder_cache', 'enabled'):
        decoder_cache = DecoderCache(
            rc.getboolean('decoder_cache', 'readonly'))
    else:
        decoder_cache = NoDecoderCache()
    return decoder_cache
