import numpy as np
import pytest

import nengo
from nengo.utils.matplotlib import rasterplot, set_color_cycle


@pytest.mark.parametrize("use_eventplot", [True, False])
def test_rasterplot(use_eventplot, Simulator, seed, plt):
    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.sin(6 * t))
        a = nengo.Ensemble(100, 1)
        nengo.Connection(u, a)
        ap = nengo.Probe(a.neurons)

    with Simulator(model) as sim:
        sim.run(1.0)

    rasterplot(sim.trange(), sim.data[ap], use_eventplot=use_eventplot)

    # TODO: add assertions


def test_rasterplot_onetime(plt):
    """Tests rasterplot with a single time point"""
    time = [0.2]
    spikes = np.array([1, 0, 0, 1, 0, 1, 1]).reshape((1, -1))
    rasterplot(time, spikes, ax=None)


def test_set_color_cycle(plt):
    if plt.__file__ == "/dev/null":
        pytest.skip("Only runs when plotting is enabled")

    slopes = np.array([-0.4, 0.3, 0.5])
    t = np.linspace(0, 1)
    y = t[:, None] * slopes

    colors1 = ["green", "blue", "red"]
    colors2 = ["red", "green", "blue"]

    # set ax1 colors with `ax` arg, and ax2 colors with `ax=None`
    ax1 = plt.subplot(211)
    set_color_cycle(colors2, ax=None)  # must be done BEFORE the axis is created
    ax2 = plt.subplot(212)

    for i in range(slopes.size):
        ax2.plot(y[:, i], label=colors2[i])
    plt.legend()

    set_color_cycle(colors1, ax=ax1)
    for i in range(slopes.size):
        ax1.plot(y[:, i], label=colors1[i])
    ax1.legend()

    for i in range(slopes.size):
        assert ax1.lines[i].get_color() == colors1[i]
        assert ax2.lines[i].get_color() == colors2[i]
