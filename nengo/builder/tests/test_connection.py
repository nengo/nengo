import numpy as np

import nengo
from nengo.builder import Model
from nengo.builder.connection import build_linear_system


def test_build_linear_system(seed, rng, plt):
    func = lambda x: x**2

    with nengo.Network(seed=seed) as net:
        conn = nengo.Connection(nengo.Ensemble(60, 1), nengo.Ensemble(50, 1),
                                function=func)

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

    assert np.allclose(Xhat, X, atol=1e-1)
    assert np.allclose(Yhat, func(X), atol=1e-1)
