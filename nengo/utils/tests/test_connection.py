import numpy as np
import pytest
import matplotlib.tri as tri

import nengo
from nengo.utils.connection import eval_point_decoding
from nengo.utils.numpy import rms


@pytest.mark.parametrize("points_arg", [False, True])
def test_eval_point_decoding(points_arg, Simulator, NonDirectNeuronType, plt, seed):
    with nengo.Network(seed=seed) as model:
        model.config[nengo.Ensemble].neuron_type = NonDirectNeuronType()
        a = nengo.Ensemble(200, 2)
        b = nengo.Ensemble(100, 1)
        c = nengo.Connection(a, b, function=lambda x: x[0] * x[1])

    kwargs = {}
    if points_arg:
        x = np.linspace(-1, 1, 51)
        y = np.linspace(-1, 1, 51)
        X, Y = np.meshgrid(x, y)
        kwargs["eval_points"] = np.column_stack([X.ravel(), Y.ravel()])

    with Simulator(model) as sim:
        eval_points, targets, decoded = eval_point_decoding(c, sim, **kwargs)

    def contour(xy, z):
        xi, yi = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
        triang = tri.Triangulation(xy[:, 0], xy[:, 1])
        interp_lin = tri.LinearTriInterpolator(triang, z.ravel())
        zi = interp_lin(xi, yi)
        plt.contourf(xi, yi, zi, cmap=plt.cm.seismic)
        plt.colorbar()

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    contour(eval_points, targets)
    plt.title("Target (desired decoding)")
    plt.subplot(132)
    plt.title("Actual decoding")
    contour(eval_points, decoded)
    plt.subplot(133)
    plt.title("Difference between actual and desired")
    contour(eval_points, decoded - targets)

    # Generous error check, just to make sure it's in the right ballpark.
    # Also make sure error is above zero, i.e. y != z
    error = rms(decoded - targets, axis=1).mean()
    assert error < 0.1 and error > 1e-8
