from distutils.version import LooseVersion
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from nengo.utils.ensemble import tuning_curves

has_prop_cycle = LooseVersion(matplotlib.__version__) >= '1.5.0'

if has_prop_cycle:
    from cycler import cycler  # Dependency of MPL form 1.5.0 onward


def get_color_cycle():
    if has_prop_cycle:
        cycle = matplotlib.rcParams['axes.prop_cycle']
        # Apparently the 'color' key may not exist, so have to fail gracefully
        try:
            return [prop['color'] for prop in cycle]
        except KeyError:
            pass  # Fall back on deprecated axes.color_cycle
    return matplotlib.rcParams['axes.color_cycle']


def set_color_cycle(colors, ax=None):
    if has_prop_cycle:
        if ax is None:
            plt.rc('axes', prop_cycle=cycler('color', colors))
        else:
            ax.set_prop_cycle('color', colors)
    else:
        if ax is None:
            plt.rc('axes', color_cycle=colors)
        else:
            ax.set_color_cycle(colors)


def axis_size(ax=None):
    """Get axis width and height in pixels.

    Based on a StackOverflow response:
    https://stackoverflow.com/questions/19306510/
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


def rasterplot(time, spikes, ax=None, use_eventplot=False, **kwargs):  # noqa
    """Generate a raster plot of the provided spike data.

    Parameters
    ----------
    time : array
        Time data from the simulation
    spikes : array
        The spike data with columns for each neuron and 1s indicating spikes
    ax : matplotlib.axes.Axes, optional
        The figure axes to plot into. If None, we will use current axes.
    use_eventplot : boolean, optional
        Whether to use the new Matplotlib `eventplot` routine. It is slower
        and makes larger image files, so we do not use it by default.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes that were plotted into

    Examples
    --------
    >>> import nengo
    >>> with nengo.Network() as net:
    ...     a = nengo.Ensemble(20, 1)
    ...     p = nengo.Probe(a.neurons)
    >>> with nengo.Simulator(net) as sim:
    ...     sim.run(1)
    >>> rasterplot(sim.trange(), sim.data[p])
    """
    n_times, n_neurons = spikes.shape

    if ax is None:
        ax = plt.gca()

    if use_eventplot and not hasattr(ax, 'eventplot'):
        warnings.warn("Matplotlib version %s does not have 'eventplot'. "
                      "Falling back to non-eventplot version."
                      % matplotlib.__version__)
        use_eventplot = False

    colors = kwargs.pop('colors', None)
    if colors is None:
        color_cycle = get_color_cycle()
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
        markersize = max(ax_height * 0.8 / n_neurons, 1)
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
        set_color_cycle(cm.to_rgba(sim.data[connection].decoders[0]), ax=ax)
    ax.plot(evals, t_curves)
