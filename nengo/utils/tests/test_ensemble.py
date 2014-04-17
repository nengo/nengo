from __future__ import absolute_import

import numpy as np
import pytest

import nengo
from nengo.utils.ensemble import tuning_curves


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


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
