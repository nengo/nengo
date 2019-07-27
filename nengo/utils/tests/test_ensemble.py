import numpy as np
import mpl_toolkits.mplot3d
import pytest

import nengo
from nengo.dists import Uniform
from nengo.utils.ensemble import response_curves, tuning_curves


def plot_tuning_curves(plt, eval_points, activities):
    if eval_points.ndim <= 2:
        plt.plot(eval_points, activities)
    elif eval_points.ndim == 3:
        assert mpl_toolkits.mplot3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(eval_points.T[0], eval_points.T[1], activities.T[0])
    else:
        raise NotImplementedError()


def test_tuning_curves_1d(Simulator, plt, seed):
    """For 1D ensembles, should be able to do plt.plot(*tuning_curves(...))."""
    model = nengo.Network(seed=seed)
    with model:
        ens_1d = nengo.Ensemble(10, dimensions=1, neuron_type=nengo.LIF())
    with Simulator(model) as sim:
        plt.plot(*tuning_curves(ens_1d, sim))


@pytest.mark.parametrize("dimensions", [1, 2])
def test_tuning_curves(Simulator, nl_nodirect, plt, seed, dimensions):
    radius = 10
    max_rate = 400
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            10,
            dimensions=dimensions,
            neuron_type=nl_nodirect(),
            max_rates=Uniform(200, max_rate),
            radius=radius,
        )
    with Simulator(model) as sim:
        eval_points, activities = tuning_curves(ens, sim)

    plot_tuning_curves(plt, eval_points, activities)

    # Check that eval_points cover up to the radius.
    assert np.abs(radius - np.max(np.abs(eval_points))) <= (2 * radius / dimensions)

    assert np.all(activities >= 0)

    d = np.sqrt(np.sum(np.asarray(eval_points) ** 2, axis=-1))
    assert np.all(activities[d <= radius] <= max_rate)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_tuning_curves_direct_mode(Simulator, plt, seed, dimensions):
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(10, dimensions, neuron_type=nengo.Direct())

    with Simulator(model) as sim:
        eval_points, activities = tuning_curves(ens, sim)

    plot_tuning_curves(plt, eval_points, activities)

    # eval_points is passed through in direct mode neurons
    assert np.allclose(eval_points, activities)


def test_response_curves(Simulator, nl_nodirect, plt, seed):
    max_rate = 400
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            10,
            dimensions=10,
            neuron_type=nl_nodirect(),
            radius=1.5,
            max_rates=Uniform(200, max_rate),
        )

    with Simulator(model) as sim:
        eval_points, activities = response_curves(ens, sim)

    plot_tuning_curves(plt, eval_points, activities)

    assert eval_points.ndim == 1 and eval_points.size > 0
    assert np.all(eval_points >= -1.0) and np.all(eval_points <= 1.0)

    assert np.all(activities >= 0.0)
    assert np.all(activities <= max_rate)
    # Activities along preferred direction must increase monotonically.
    assert np.all(np.diff(activities, axis=0) >= 0.0)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_response_curves_direct_mode(Simulator, plt, seed, dimensions):
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            10, dimensions=dimensions, neuron_type=nengo.Direct(), radius=1.5
        )

    with Simulator(model) as sim:
        eval_points, activities = response_curves(ens, sim)

    plot_tuning_curves(plt, eval_points, activities)

    assert eval_points.ndim == 1 and eval_points.size > 0
    assert np.all(eval_points >= -1.0) and np.all(eval_points <= 1.0)
    # eval_points is passed through in direct mode neurons
    assert np.allclose(eval_points, activities)
