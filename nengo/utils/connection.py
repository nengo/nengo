import numpy as np

import nengo


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
    # pylint: disable=import-outside-toplevel
    # note: these are imported here to avoid circular imports
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
