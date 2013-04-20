import atexit
import os
import shelve
import tempfile


def generate_ensemble_key(neurons, dimensions, tau_rc, tau_ref, max_rate,
                          intercept, radius, encoders, decoder_noise,
                          eval_points, noise, seed, dt):

    key = '%d_%d_%g_%g' % (neurons, dimensions, tau_rc, tau_ref)

    if type(max_rate) is tuple and len(max_rate) == 2:
        key += '_%1.1f_%1.1f' % max_rate
    else:
        key += '_%08x' % hash(tuple(max_rate))

    if type(intercept) is tuple and len(intercept) == 2:
        key += '_%g_%g' % intercept
    else:
        key += '_%08x' % hash(tuple(intercept))

    key += '_%g_%g_%g' % (radius, decoder_noise, dt)

    # TODO: use some approach other than hoping that this hash
    # does not have collisions
    if encoders is not None:
        key += '_enc%08x' % hash(tuple([tuple(x) for x in encoders]))
    if eval_points is not None:
        key += '_eval%08x' % hash(tuple([tuple(x) for x in eval_points]))
    if seed is not None:
        key += '_seed%08x' % seed

    return key


def get_gamma_inv(key):
    return cache.get(key, None)


def set_gamma_inv(key, value):
    cache[key] = value

cache = shelve.open(os.path.join(tempfile.gettempdir(),
                                 'nefpy_cache_gamma_inv'))
atexit.register(cache.close)
