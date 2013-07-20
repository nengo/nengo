import numpy as np

from nengo.objects import LIF

def test_lif_rate_basic():
    """Test that the rate model approximately matches the
    dynamic model.

    """
    lif = LIF(10)
    J = np.arange(0, 5, .5)
    voltage = np.zeros(10)
    reftime = np.zeros(10)

    spikes = []
    spikes_ii = np.zeros(10)

    for ii in range(1000):
        lif.step_math0(.001, J + lif.bias, voltage, reftime, spikes_ii)
        spikes.append(spikes_ii.copy())

    sim_rates = np.sum(spikes, axis=0)

    math_rates = lif.rates(J)

    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)

