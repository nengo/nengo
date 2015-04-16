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


def eval_point_decoding(conn, sim, eval_points=None):
    """Get the targets and actual decoded values for a set of eval points.

    This function evaluates the static decoding (i.e. using the neuron type's
    `rates` function) of a connection for a given set of evaluation points.

    Parameters
    ----------
    conn : Connection
        The Connection to evaluate the decoding of.
    sim : Simulator
        A Nengo simulator storing the built connection.
    eval_points : array_like (N, E) (optional)
        An N x E array of evaluation points to evaluate the decoding for, where
        N is the number of points and E is the dimensionality of the input
        ensemble (i.e. `conn.size_in`). If None (default), use the connection's
        training evaluation points.

    Returns
    -------
    eval_points : ndarray (N, E)
        A shallow copy of the evaluation points used. E is the dimensionality
        of the connection input ensemble (i.e. `conn.size_in`).
    targets : ndarray (N, D)
        The target function value at each evaluation point.
    decoded : ndarray (N, D)
        The decoded function value at each evaluation point.
    """
    from nengo.builder.ensemble import get_activities
    from nengo.builder.connection import get_targets

    if eval_points is None:
        eval_points = sim.data[conn].eval_points
    else:
        eval_points = np.asarray(eval_points)

    decoders = sim.data[conn].decoders
    if decoders is None:
        raise ValueError("Connection must have decoders")

    activities = get_activities(sim.model, conn.pre_obj, eval_points)
    decoded = np.dot(activities, decoders.T)
    targets = get_targets(sim.model, conn, eval_points)
    return eval_points, targets, decoded
