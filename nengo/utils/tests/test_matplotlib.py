import logging

import numpy as np
import pytest

import nengo

logger = logging.getLogger(__name__)


@pytest.mark.plot
@pytest.mark.parametrize('use_eventplot', [True, False])
def test_rasterplot(use_eventplot, Simulator, seed, plt):
    from nengo.utils.matplotlib import rasterplot

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.sin(6 * t))
        a = nengo.Ensemble(100, 1)
        nengo.Connection(u, a)
        ap = nengo.Probe(a.neurons)

    sim = Simulator(model)
    sim.run(1.0)

    rasterplot(sim.trange(), sim.data[ap], use_eventplot=use_eventplot)
    if use_eventplot:
        plt.saveas = 'utils.test_matplotlib.test_rasterplot.eventplot.pdf'
