import hashlib
import os
import os.path
import struct

import numpy as np


class DecoderCache(object):
    _FILE_EXTENSION = '.npy'

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def wrap_solver(self, solver):
        # TODO can the default arguments be copied from the function wrapped?
        # This shouldn't use *args and **kwargs, because this should raise an
        # exception for arguments which do not get hashed and might lead to
        # wrong cache hits.
        def cached_solver(activities, targets, rng=np.random, E=None):
            key = self._get_cache_key(solver, activities, targets, rng, E)
            path = self._get_path(key)
            if os.path.exists(path):
                # TODO log hit
                # FIXME no solver_info restored
                return np.load(path), {}
            else:
                # TODO log miss
                # FIXME no solver_info stored
                decoders, solver_info = solver(
                    activities, targets, rng=rng, E=E)
                np.save(path, decoders)
                return decoders, solver_info
        return cached_solver

    def _get_cache_key(self, solver, activities, targets, rng, E):
        h = hashlib.sha1()

        h.update(solver.__module__.encode())
        h.update(solver.__name__.encode())

        h.update(activities.data)
        h.update(targets.data)

        # rng format doc:
        # noqa <http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.get_state.html#numpy.random.RandomState.get_state>
        state = rng.get_state()
        h.update(state[0].encode())  # string 'MT19937'
        h.update(state[1].data)  # 1-D array of 624 unsigned integer keys
        h.update(struct.pack('q', state[2]))  # integer pos
        h.update(struct.pack('q', state[3]))  # integer has_gauss
        h.update(struct.pack('d', state[4]))  # float cached_gaussian

        if E is not None:
            h.update(E.data)
        return h.hexdigest()

    def _get_path(self, key):
        return os.path.join(self.cache_dir, key + self._FILE_EXTENSION)
