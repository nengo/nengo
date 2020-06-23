import numpy as np
import pytest

import nengo
from nengo.builder import Model
from nengo.builder.connection import build_linear_system
from nengo.exceptions import BuildError


def test_build_linear_system(seed, rng, plt, allclose):
    func = lambda x: x ** 2

    with nengo.Network(seed=seed) as net:
        conn = nengo.Connection(
            nengo.Ensemble(60, 1), nengo.Ensemble(50, 1), function=func
        )

    model = Model()
    model.build(net)
    eval_points, activities, targets = build_linear_system(model, conn, rng)
    assert eval_points.shape[1] == 1
    assert targets.shape[1] == 1
    assert activities.shape[1] == 60
    assert eval_points.shape[0] == activities.shape[0] == targets.shape[0]

    X = eval_points
    AA = activities.T.dot(activities)
    AX = activities.T.dot(eval_points)
    AY = activities.T.dot(targets)
    WX = np.linalg.solve(AA, AX)
    WY = np.linalg.solve(AA, AY)

    Xhat = activities.dot(WX)
    Yhat = activities.dot(WY)

    i = np.argsort(eval_points.ravel())
    plt.plot(X[i], Xhat[i])
    plt.plot(X[i], Yhat[i])

    assert allclose(Xhat, X, atol=1e-1)
    assert allclose(Yhat, func(X), atol=1e-1)


def test_build_linear_system_zeroact(seed, rng):
    eval_points = np.linspace(-0.1, 0.1, 100)[:, None]

    with nengo.Network(seed=seed) as net:
        a = nengo.Ensemble(5, 1, intercepts=nengo.dists.Choice([0.9]))
        b = nengo.Ensemble(5, 1, intercepts=nengo.dists.Choice([0.9]))

    model = Model()
    model.build(net)

    conn = nengo.Connection(a, b, eval_points=eval_points, add_to_container=False)
    with pytest.raises(BuildError, match="'activities' matrix is all zero"):
        build_linear_system(model, conn, rng)


def test_build_connection_errors():
    # --- test function points on direct connection error
    with nengo.Network() as net:
        a = nengo.Ensemble(5, 1, neuron_type=nengo.Direct())
        b = nengo.Ensemble(4, 1)
        nengo.Connection(a, b, eval_points=np.ones((3, 1)), function=np.ones((3, 1)))

    with pytest.raises(BuildError, match="Cannot use function points.*direct conn"):
        with nengo.Simulator(net):
            pass

    # --- test connection to object not in the model
    b = nengo.Ensemble(5, 1, add_to_container=False)
    with nengo.Network() as net:
        a = nengo.Ensemble(5, 1)
        conn = nengo.Connection(a, b)

    with pytest.raises(BuildError, match="is not in the model"):
        with nengo.Simulator(net):
            pass

    # --- test connection with post object of size zero
    with nengo.Network() as net:
        a = nengo.Ensemble(5, 1)
        b = nengo.Node([0])
        conn = nengo.Connection(a, a)

    # hack to get around API validation
    nengo.Connection.post.data[conn] = b

    with pytest.raises(BuildError, match="the 'post' object.*has a 'in' size of zero"):
        with nengo.Simulator(net):
            pass
