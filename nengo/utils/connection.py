from __future__ import absolute_import

import warnings

import numpy as np

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
    warnings.warn("'targets' can be passed directly to the connection through "
                  "the 'function' argument. That approach is faster, so this "
                  "function is deprecated and will be removed in the future.")

    eval_points = npext.array(eval_points, dtype=np.float64, min_dims=2)
    targets = npext.array(targets, dtype=np.float64, min_dims=2)

    if len(eval_points) != len(targets):
        raise ValidationError(
            "Number of evaluation points (%d) is not equal to the number of "
            "targets (%s)" % (len(eval_points), len(targets)), 'eval_points')

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

    weights = sim.data[conn].weights
    activities = get_activities(sim.model, conn.pre_obj, eval_points)
    decoded = np.dot(activities, weights.T)
    targets = get_targets(sim.model, conn, eval_points)
    return eval_points, targets, decoded


def pes_learning_rate(epsilon, activities, t, dt=0.001):
    """Determine the ideal learning rate for PES without noise or filtering.

    This function returns a `learning_rate` for use in the PES rule, such that
    after `t` seconds (with a simulator timestep of `dt`) a constant input
    will have error equal to `epsilon` times the initial error. [1]_

    Parameters
    ----------
    epsilon : float
        The desired approximation factor. The resulting error will be `epsilon`
        times the initial error. If you want the error to be at most some
        constant, then divide `epsilon` by the largest possible initial error
        (usually no more than 2, when the radius is 1).
    activities : array_like (N,)
        An array of N activity rates. Less activity (small ||a||) need a higher
        learning rate. Pick the activities with the smallest ||a|| that you
        want to learn within epsilon, or make it the average firing rate of
        each neuron.
    t : float
        The amount of simulation time (in seconds) required to obtain the
        desired error.
    dt : float (optional)
        The simulation timestep, defaults to 1 ms.

    Returns
    -------
    learning_rate : float
        The learning rate to provide to the PES rule.
    gamma : float
        The rate of convergence, such that the error is the initial error
        multiplied by `gamma ** k` on the k'th timestep.

    References
    ----------
    .. [1] http://compneuro.uwaterloo.ca/publications/voelker2015.html
    """
    activities = np.asarray(activities)
    if activities.ndim != 1:
        raise ValueError("activities must be a one-dimensional array")
    n, = activities.shape  # number of neurons
    a_sq = np.dot(activities.T, activities) * dt  # ||a||^2
    k = (t - dt) / dt  # number of simulation timesteps
    gamma = epsilon**(1.0 / k)  # rate of convergence
    kappa = (1 - gamma) / a_sq  # rearrange equation from theorem
    return kappa * n, gamma
