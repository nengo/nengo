import warnings

import numpy as np

import nengo
from . import numpy as npext
from ..exceptions import ValidationError


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
       A dictionary with two keys: ``function`` and ``eval_points``.
       function is the mapping between the evaluation points and the
       targets. ``eval_points`` are the evaluation points that will
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
    warnings.warn(
        "'targets' can be passed directly to the connection through "
        "the 'function' argument. That approach is faster, so this "
        "function is deprecated and will be removed in the future."
    )

    dtype = nengo.rc.float_dtype
    eval_points = npext.array(eval_points, dtype=dtype, min_dims=2)
    targets = npext.array(targets, dtype=dtype, min_dims=2)

    if len(eval_points) != len(targets):
        raise ValidationError(
            "Number of evaluation points (%d) is not equal to the number of "
            "targets (%s)" % (len(eval_points), len(targets)),
            "eval_points",
        )

    func_dict = {}
    for eval_point, target in zip(eval_points, targets):
        func_dict[tuple(eval_point)] = target

    def function(x):
        x = tuple(x)
        return func_dict[x]

    return {
        "function": function,
        "eval_points": eval_points,
        "scale_eval_points": False,
    }


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

    dtype = nengo.rc.float_dtype

    if eval_points is None:
        eval_points = sim.data[conn].eval_points
    else:
        eval_points = np.asarray(eval_points, dtype=dtype)

    ens = conn.pre_obj
    weights = sim.data[conn].weights
    activities = get_activities(sim.data[ens], ens, eval_points)
    decoded = np.dot(activities, weights.T)
    targets = get_targets(conn, eval_points, dtype=dtype)
    return eval_points, targets, decoded
