

import numpy as np
import matplotlib.pyplot as plt


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
    >>> plt.figure()
    >>> rasterplot(sim.data(sim.model.t), sim.data('A.spikes'))
    '''

    if ax is None:
        ax = plt.gca()

    colors = kwargs.pop('colors', None)
    if colors is None:
        color_cycle = ax._get_lines.color_cycle
        colors = [next(color_cycle) for _ in range(spikes.shape[1])]

    if hasattr(ax, 'eventplot'):
        spikes = [time[spikes[:,i] > 0].flatten()
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
            ax.plot(time[spikes[:,i] > 0],
                    np.ones_like(np.where(spikes[:,i] > 0)).T + i, ',',
                    color=colors[i], **kwargs)

    return ax
