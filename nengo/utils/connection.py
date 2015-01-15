from __future__ import absolute_import

import numpy as np

from . import numpy as npext


def target_function(eval_points, targets):
    """Get a function that maps evaluation points to target points.

    Use this when making a nengo connection using a sequence
    of evaluation points and targets, instead of passing a
    callable as the connection function.

    Parameters
    ----------
    eval_points: iterable
        A sequence of evaluation points.
    targets: iterable
        A sequence of targets with the same length as ``eval_points``

    Returns
    -------
    dict:
       A diciontary with two keys: ``function`` and ``eval_points``.
       function is the mapping between the evaluation points and the
       targets. ``eval_points`` are the evalutaion points that will
       be passed to the connection

    Examples
    --------
    ens1 = nengo.Ensemble(n_neurons=100, dimensions=1)
    ens2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    eval_points = numpy.arange(-1, 1, 0.01)
    targets = numpy.sin(eval_points)
    #the transformation on this connection approximates a sin function
    nengo.Connection(ens1, ens2,
                     **target_function(eval_points, targets)
    """

    eval_points = npext.array(eval_points, dtype=np.float64, min_dims=2)
    targets = npext.array(targets, dtype=np.float64, min_dims=2)

    if len(eval_points) != len(targets):
        raise ValueError("Number of evaluation points %s "
                         "is not equal to number of targets "
                         "%s" % (len(eval_points), len(targets)))

    func_dict = {}
    for eval_point, target in zip(eval_points, targets):
        func_dict[tuple(eval_point)] = target

    def function(x):
        x = tuple(x)
        return func_dict[x]

    return {'function': function,
            'eval_points': eval_points,
            'scale_eval_points': False}
