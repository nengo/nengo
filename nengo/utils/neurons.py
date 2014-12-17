from __future__ import absolute_import
import logging

import numpy as np

from . import numpy as npext

logger = logging.getLogger(__name__)


def spikes2events(t, spikes):
    """Return an event-based representation of spikes (i.e. spike times)"""
    spikes = npext.array(spikes, copy=False, min_dims=2)
    if spikes.ndim > 2:
        raise ValueError("Cannot handle %d-dimensional arrays" % spikes.ndim)
    if spikes.shape[-1] != len(t):
        raise ValueError("Last dimension of `spikes` must equal `len(t)`")

    # find nonzero elements (spikes) in each row, and translate to times
    return [t[spike != 0] for spike in spikes]


def _rates_isi_events(t, events, midpoint, interp):
    import scipy.interpolate

    if len(events) == 0:
        return np.zeros_like(t)

    isis = np.diff(events)

    rt = np.zeros(len(events) + (1 if midpoint else 2))
    rt[1:-1] = 0.5*(events[:-1] + events[1:]) if midpoint else events
    rt[0], rt[-1] = t[0], t[-1]

    r = np.zeros_like(rt)
    r[1:len(isis) + 1] = 1. / isis

    f = scipy.interpolate.interp1d(rt, r, kind=interp, copy=False)
    return f(t)


def rates_isi(t, spikes, midpoint=False, interp='zero'):
    """Estimate firing rates from spikes using ISIs.

    Parameters
    ----------
    t : (M,) array_like
        The times at which raw spike data (spikes) is defined.
    spikes : (M, N) array_like
        The raw spike data from N neurons.
    midpoint : bool, optional
        If true, place interpolation points at midpoints of ISIs. Otherwise,
        the points are placed at the beginning of ISIs.
    interp : string, optional
        Interpolation type, passed to `scipy.interpolate.interp1d` as the
        `kind` parameter.

    Returns
    -------
    rates : (M, N) array_like
        The estimated neuron firing rates.
    """
    spike_times = spikes2events(t, spikes.T)
    rates = np.zeros(spikes.shape)
    for i, st in enumerate(spike_times):
        rates[:, i] = _rates_isi_events(t, st, midpoint, interp)

    return rates


def lowpass_filter(x, tau, kind='expon'):
    nt = x.shape[-1]

    if kind == 'expon':
        t = np.arange(0, 5 * tau)
        kern = np.exp(-t / tau) / tau
        delay = tau
    elif kind == 'gauss':
        std = tau / 2.
        t = np.arange(-4 * std, 4 * std)
        kern = np.exp(-0.5 * (t / std)**2) / np.sqrt(2 * np.pi * std**2)
        delay = 4 * std
    elif kind == 'alpha':
        alpha = 1. / tau
        t = np.arange(0, 5 * tau)
        kern = alpha**2 * t * np.exp(-alpha * t)
        delay = tau
    else:
        raise ValueError("Unrecognized filter kind '%s'" % kind)

    delay = int(np.round(delay))
    return np.array(
        [np.convolve(kern, xx, mode='full')[delay:nt + delay] for xx in x])


def rates_kernel(t, spikes, kind='gauss', tau=0.04):
    """Estimate firing rates from spikes using a kernel.

    Parameters
    ----------
    t : (M,) array_like
        The times at which raw spike data (spikes) is defined.
    spikes : (M, N) array_like
        The raw spike data from N neurons.
    kind : str {'expon', 'gauss', 'expogauss', 'alpha'}, optional
        The type of kernel to use. 'expon' is an exponential kernel, 'gauss' is
        a Gaussian (normal) kernel, 'expogauss' is an exponential followed by
        a Gaussian, and 'alpha' is an alpha function kernel.
    tau : float
        The time constant for the kernel. The optimal value will depend on the
        firing rate of the neurons, with a longer tau preferred for lower
        firing rates. The default value of 0.04 works well across a wide range
        of firing rates.
    """
    spikes = spikes.T
    spikes = npext.array(spikes, copy=False, min_dims=2)
    if spikes.ndim > 2:
        raise ValueError("Cannot handle %d-dimensional arrays" % spikes.ndim)
    if spikes.shape[-1] != len(t):
        raise ValueError("Last dimension of `spikes` must equal `len(t)`")

    n, nt = spikes.shape
    dt = t[1] - t[0]

    tau_i = tau / dt
    kind = kind.lower()
    if kind == 'expogauss':
        rates = lowpass_filter(spikes, tau_i, kind='expon')
        rates = lowpass_filter(rates, tau_i / 4, kind='gauss')
    else:
        rates = lowpass_filter(spikes, tau_i, kind=kind)

    return rates.T


def settled_firingrate(step_math, J, states,
                       dt=0.001, settle_time=0.1, sim_time=1.0):
    """Compute firing rates (in Hz) for given vector input, ``x``.

    Unlike the default naive implementation, this approach takes into
    account some characteristics of spiking neurons. We start
    by simulating the neurons for a short amount of time, to let any
    initial transients settle. Then, we run the neurons for a second
    and find the average (which should approximate the firing rate).

    Parameters
    ---------
    step_math : function
        the step function of the neuron type
    J : ndarray
        a vector of currents to generate firing rates from
    *states : list of ndarrays
        additional state needed by the step function
    """
    out = np.zeros_like(J)
    total = np.zeros_like(J)

    # Simulate for the settle time
    steps = int(settle_time / dt)
    for _ in range(steps):
        step_math(dt, J, out, *states)
    # Simulate for sim time, and keep track
    steps = int(sim_time / dt)
    for _ in range(steps):
        step_math(dt, J, out, *states)
        total += out
    return total / float(steps)
