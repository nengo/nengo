from __future__ import absolute_import

import warnings

import numpy as np
import matplotlib.pyplot as plt

from nengo.utils.compat import range
from nengo.utils.ensemble import tuning_curves


def implot(plt, x, y, Z, ax=None, colorbar=True, **kwargs):
    """
    Image plot of general data (like imshow but with non-pixel axes).

    Parameters
    ----------
    plt : plot object
        Plot object, typically `matplotlib.pyplot`.
    x : (M,) array_like
        Vector of x-axis points, must be linear (equally spaced).
    y : (N,) array_like
        Vector of y-axis points, must be linear (equally spaced).
    Z : (M, N) array_like
        Matrix of data to be displayed, the value at each (x, y) point.
    ax : axis object (optional)
        A specific axis to plot on (defaults to `plt.gca()`).
    colorbar: boolean (optional)
        Whether to plot a colorbar.
    **kwargs
        Additional arguments for `ax.imshow`.
    """
    ax = plt.gca() if ax is None else ax

    def is_linear(x):
        diff = np.diff(x)
        return np.allclose(diff, diff[0])

    assert is_linear(x) and is_linear(y)
    image = ax.imshow(Z, aspect='auto', extent=(x[0], x[-1], y[-1], y[0]),
                      **kwargs)
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
        ax.set_xlim(left=0, right=max(time))

    else:
        # Older Matplotlib, doesn't have eventplot
        for i in range(spikes.shape[1]):
            ax.plot(time[spikes[:, i] > 0],
                    np.ones_like(np.where(spikes[:, i] > 0)).T + i, ',',
                    color=colors[i], **kwargs)

    return ax


def plot_tuning_curves(ensemble, sim, connection=None, ax=None):
    """Plot tuning curves for the given ensemble and simulator.

    If a connection is provided, the decoders will be used to set
    the colours of the tuning curves.
    """

    if ax is None:
        ax = plt.gca()

    evals, t_curves = tuning_curves(ensemble, sim)

    if connection is not None:
        if connection.dimensions > 1:
            warnings.warn("Ignoring dimensions > 1 in plot_tuning_curves")
        cm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        ax.set_color_cycle(cm.to_rgba(sim.data[connection].decoders[0]))
    ax.plot(evals, t_curves)
