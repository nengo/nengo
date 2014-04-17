from __future__ import absolute_import

import numpy as np
import pytest

import nengo
from nengo.utils.ensemble import (
    tuning_curves, tuning_curves_along_pref_direction)


@pytest.mark.parametrize('dimensions', [1, 2])
def test_tuning_curves(Simulator, plt, seed, dimensions):
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(10, dimensions, neuron_type=nengo.Direct())
    sim = Simulator(model)

    eval_points, activities = tuning_curves(ens, sim)

    plt.plot(eval_points, activities)
    plt.saveas = ('utils.test_ensemble.test_tuning_curves_direct_mode_%d.pdf'
                  % dimensions)

    # eval_points is passed through in direct mode neurons
    assert np.allclose(eval_points, activities)


def test_tuning_curves_along_pref_direction(Simulator, seed, plt):
    model = nengo.Network(seed=seed)
    with model:
        ens = nengo.Ensemble(
            30, dimensions=10, radius=1.5, neuron_type=nengo.Direct())
    sim = Simulator(model)

    x, activities = tuning_curves_along_pref_direction(ens, sim)

    plt.plot(x, activities)

    assert x.ndim == 1 and x.size > 0
    assert np.all(-1.5 <= x) and np.all(x <= 1.5)
    # eval_points is passed through in direct mode neurons
    assert np.allclose(x, activities)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
