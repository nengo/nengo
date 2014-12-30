from __future__ import absolute_import

import warnings

import numpy as np
import matplotlib.pyplot as plt

from nengo.utils.compat import range
from nengo.utils.ensemble import tuning_curves


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


def rasterplot(time, spikes, ax=None, use_eventplot=None, **kwargs):  # noqa: C901
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
        Whether to use the new Matplotlib `eventplot` routine. By default,
        we use the routine if it exists (newer Matplotlib).

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
    if use_eventplot is None:
        use_eventplot = has_eventplot

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
        kwargs.setdefault('markersize', 3)  # looks good for 100 neurons
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
