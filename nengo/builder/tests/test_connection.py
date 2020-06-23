import numpy as np

import pytest
import nengo
from nengo.exceptions import BuildError
from nengo.builder import Model
from nengo.neurons import Direct
from nengo.builder.connection import build_linear_system, build_connection


def test_build_errors_1(seed, rng):
    """tests the build errors"""

    func = lambda x: x ** 2

    with nengo.Network(seed=seed) as net:
        conn = nengo.Connection(
            nengo.Ensemble(60, 1), nengo.Ensemble(50, 1), function=func
        )

    model = Model()
    model.build(net)

    class fakeReturn:
        def myFunction2(self, write=False):
            return True

        def __mul__(self, other):
            return 1

        def __len__(self):
            return 1

        def __getitem__(self, other):
            return [0]

        dtype = "float64"

        setflags = myFunction2

    class fakeEval:

        view = fakeReturn

    class fakeEncoders:
        T = False

    class Test:
        bias = 0
        gain = 0
        encoders = fakeEncoders()
        eval_points = fakeEval

    model.params[conn.pre_obj] = Test()

    # Building %s: 'activites' matrix is all zero for %s.
    # with pytest.raises(BuildError):

    with pytest.raises(BuildError):
        conn.eval_points = np.ndarray((1, 1))
        conn.function = np.ndarray((1, 1))

        conn.pre_obj.neuron_type = Direct()
        build_connection(model, conn)

    with pytest.raises(BuildError):
        model.sig[conn.pre_obj] = []
        build_connection(model, conn)

    with nengo.Network(seed=seed) as net:
        conn = nengo.Connection(
            nengo.Ensemble(60, 1), nengo.Ensemble(50, 1), function=func
        )

    model = Model()
    model.build(net)

    with nengo.Network() as net:
        ens = nengo.Ensemble(1, 1, gain=np.array([0]), bias=np.array([-10]))
        conn = nengo.Connection(ens, ens)
        # What is this code doing?

        model.params[conn.pre_obj] = Test()

        conn.eval_points = [[0]]

        with pytest.raises(AssertionError):
            build_linear_system(model, conn, rng)


def test_build_errors_2(Simulator, seed, rng, plt, allclose):
    n = 200

    m = nengo.Network(seed=seed)
    with m:
        u = nengo.Node((1, 2), size_out=2)
        a = nengo.Ensemble(n, dimensions=2)
        b = nengo.Ensemble(n, dimensions=2)
        nengo.Connection(u, a)

        nengo.Connection(
            a.neurons, b.neurons, transform=1, learning_rule_type=nengo.BCM()
        )

    with pytest.raises(BuildError):
        with Simulator(m):
            pass


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
