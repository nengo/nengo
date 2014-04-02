from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt


def implot(plt, x, y, Z, ax=None, colorbar=True, **kwargs):
    ax = plt.gca() if ax is None else ax

    def is_linear(x):
        diff = np.diff(x)
        return np.allclose(diff, diff[0])

    assert is_linear(x) and is_linear(y)
    image = ax.imshow(Z, aspect='auto', extent=(x[0], x[-1], y[-1], y[0]))
    if colorbar:
        plt.colorbar(image, ax=ax)


def rasterplot(time, spikes, ax=None, **kwargs):
    '''Generate a raster plot of the provided spike data

    Parameters
    ----------
    time : array
        Time data from the simulation
    spikes: array
        The spike data with columns for each neuron and 1s indicating spikes
    ax: matplotlib.axes.Axes
        The figure axes to plot into.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axes that were plotted into

    Examples
    --------
    >>> import nengo
    >>> model = nengo.Model("Raster")
    >>> A = nengo.Ensemble(nengo.LIF(20), dimensions=1)
    >>> A_spikes = nengo.Probe(A, "spikes")
    >>> sim = nengo.Simulator(model)
    >>> sim.run(1)
    >>> rasterplot(sim.trange(), sim.data[A_spikes])
    '''

    if ax is None:
        ax = plt.gca()

    colors = kwargs.pop('colors', None)
    if colors is None:
        color_cycle = plt.rcParams['axes.color_cycle']
        colors = [color_cycle[ix % len(color_cycle)]
                  for ix in range(spikes.shape[1])]

    if hasattr(ax, 'eventplot'):
        spikes = [time[spikes[:, i] > 0].flatten()
                  for i in range(spikes.shape[1])]
        for ix in range(len(spikes)):
            if spikes[ix].shape == (0,):
                spikes[ix] = np.array([-1])
        ax.eventplot(spikes, colors=colors, **kwargs)
        ax.set_ylim(len(spikes) - 0.5, -0.5)
        if len(spikes) == 1:
            ax.set_ylim(0.4, 1.6)  # eventplot plots different for len==1
        ax.set_xlim(left=0)

    else:
        # Older Matplotlib, doesn't have eventplot
        for i in range(spikes.shape[1]):
            ax.plot(time[spikes[:, i] > 0],
                    np.ones_like(np.where(spikes[:, i] > 0)).T + i, ',',
                    color=colors[i], **kwargs)

    return ax
