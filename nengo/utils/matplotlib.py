from __future__ import absolute_import

import warnings

import numpy as np
import matplotlib.pyplot as plt

from nengo.utils.compat import range
from nengo.utils.ensemble import tuning_curves


def axis_size(ax=None):
    """Get axis width and height in pixels.

    Based on a StackOverflow response:
    http://stackoverflow.com/questions/19306510/
        determine-matplotlib-axis-size-in-pixels

    Parameters
    ----------
    ax : axis object
        The axes to determine the size of. Defaults to current axes.

    Returns
    -------
    width : float
        Width of axes in pixels.
    height : float
        Height of axes in pixels.
    """
    ax = plt.gca() if ax is None else ax
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return bbox.width * fig.dpi, bbox.height * fig.dpi


def implot(plt, x, y, Z, ax=None, colorbar=True, **kwargs):
    """Image plot of general data (like imshow but with non-pixel axes).

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


def rasterplot(time, spikes, ax=None, use_eventplot=False, **kwargs):  # noqa: C901
    """Generate a raster plot of the provided spike data

    Parameters
    ----------
    time : array
        Time data from the simulation
    spikes: array
        The spike data with columns for each neuron and 1s indicating spikes
    ax: matplotlib.axes.Axes
        The figure axes to plot into.
    use_eventplot: boolean
        Whether to use the new Matplotlib `eventplot` routine. It is slower
        and makes larger image files, so we do not use it by default.

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
    """
    n_times, n_neurons = spikes.shape

    if ax is None:
        ax = plt.gca()

    # older Matplotlib doesn't have eventplot
    has_eventplot = hasattr(ax, 'eventplot')
    if use_eventplot and not has_eventplot:
        raise ValueError("Your Matplotlib version does not have 'eventplot'")

    colors = kwargs.pop('colors', None)
    if colors is None:
        color_cycle = plt.rcParams['axes.color_cycle']
        colors = [color_cycle[i % len(color_cycle)] for i in range(n_neurons)]

    # --- plotting
    if use_eventplot:
        spiketimes = [time[s > 0].ravel() for s in spikes.T]
        for ix in range(n_neurons):
            if spiketimes[ix].size == 0:
                spiketimes[ix] = np.array([-np.inf])

        # hack to make 'eventplot' count from 1 instead of 0
        spiketimes = [np.array([-np.inf])] + spiketimes
        colors = [['k']] + colors

        ax.eventplot(spiketimes, colors=colors, **kwargs)
    else:
        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('marker', '|')
        # Default markersize determined by matching eventplot
        ax_height = axis_size(ax)[1]
        markersize = max(ax_height * 0.965 / n_neurons, 1)
        # For 1 - 3 neurons, we need an extra fudge factor to match eventplot
        markersize -= max(4 - n_neurons, 0) ** 2 * ax_height * 0.005
        kwargs.setdefault('markersize', markersize)
        kwargs.setdefault('markeredgewidth', 1)

        for i in range(n_neurons):
            spiketimes = time[spikes[:, i] > 0].ravel()
            ax.plot(spiketimes, np.zeros_like(spiketimes) + (i + 1),
                    color=colors[i], **kwargs)

    # --- set axes limits
    if n_times > 1:
        ax.set_xlim(time[0], time[-1])

    ax.set_ylim(n_neurons + 0.6, 0.4)
    if n_neurons < 5:
        # make sure only integer ticks for small neuron numbers
        ax.set_yticks(np.arange(1, n_neurons + 1))

    # --- remove ticks as these are distracting in rasters
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

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
        cm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
        ax.set_color_cycle(cm.to_rgba(sim.data[connection].decoders[0]))
    ax.plot(evals, t_curves)
