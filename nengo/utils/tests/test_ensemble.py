from __future__ import absolute_import

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
import pytest

import nengo
from nengo.utils.distributions import Uniform
from nengo.utils.ensemble import response_curves, tuning_curves


def plot_tuning_curves(plt, eval_points, activities):
    if len(eval_points) == 1:
        plt.plot(eval_points[0], activities.T)
    elif len(eval_points) == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(eval_points[0], eval_points[1], activities[0])
    else:
        raise NotImplementedError()


@pytest.mark.parametrize('dimensions', [1, 2])
def test_tuning_curves(Simulator, plt, seed, dimensions):
    max_rate = 400
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            10, dimensions=dimensions, neuron_type=nengo.LIF(),
            max_rates=Uniform(200, max_rate))
    sim = Simulator(model)

    eval_points, activities = tuning_curves(ens, sim)

    plt.saveas = 'utils.test_ensemble.test_tuning_curves_%d.pdf' % dimensions
    plot_tuning_curves(plt, eval_points, activities)

    assert np.all(activities >= 0)
    # Activity might be larger than max_rate as evaluation points will be taken
    # outside the ensemble radius.
    assert np.all(activities <= max_rate * np.sqrt(dimensions))


@pytest.mark.parametrize('dimensions', [1, 2])
def test_tuning_curves_direct_mode(Simulator, plt, seed, dimensions):
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(10, dimensions, neuron_type=nengo.Direct())
    sim = Simulator(model)

    eval_points, activities = tuning_curves(ens, sim)

    plt.saveas = ('utils.test_ensemble.test_tuning_curves_direct_mode_%d.pdf'
                  % dimensions)
    plot_tuning_curves(plt, eval_points, activities)

    # eval_points is passed through in direct mode neurons
    assert np.allclose(eval_points, activities)


def test_response_curves(Simulator, plt, seed):
    max_rate = 400
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            10, dimensions=10, neuron_type=nengo.LIF(), radius=1.5,
            max_rates=Uniform(200, max_rate))
    sim = Simulator(model)

    eval_points, activities = response_curves(ens, sim)
    plt.plot(eval_points, activities)

    assert eval_points.ndim == 1 and eval_points.size > 0
    assert np.all(-1.0 <= eval_points) and np.all(eval_points <= 1.0)

    assert np.all(activities >= 0.0)
    assert np.all(activities <= max_rate)
    # Activities along preferred direction must increase monotonically.
    assert np.all(np.diff(activities, axis=0) >= 0.0)


@pytest.mark.parametrize('dimensions', [1, 2])
def test_response_curves_direct_mode(Simulator, plt, seed, dimensions):
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            10, dimensions=dimensions, neuron_type=nengo.Direct(), radius=1.5)
    sim = Simulator(model)

    eval_points, activities = response_curves(ens, sim)

    plt.saveas = ('utils.test_ensemble.test_response_curves_direct_mode_%d.pdf'
                  % dimensions)
    plt.plot(eval_points, activities)

    assert eval_points.ndim == 1 and eval_points.size > 0
    assert np.all(-1.0 <= eval_points) and np.all(eval_points <= 1.0)
    # eval_points is passed through in direct mode neurons
    assert np.allclose(eval_points, activities)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
