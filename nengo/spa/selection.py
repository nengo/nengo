"""Selection networks."""

import nengo
import numpy as np
from nengo.utils.compat import is_iterable


def LinearFunc():
    """Returns a linear mapping function.

    Returns a linear mapping function for ease of use when creating output
    mappings. The returned function takes in a non-optional 'x' parameter.
    """
    return lambda x: x


def FilteredStepFunc(x_scale=15.0):
    """Returns a filtered step mapping function.

    Returns a filtered step mapping function for ease of use when creating
    output mappings. The returned function takes in a non-optional 'x'.

    Parameters
    ----------
    x_scale: float, optional
        Scaling factor to be applied to the 'x' input. Affects the
        sharpness of the filtered step edge. Larger values produce a
        sharper edge. Set to a negative value to flip the mapping
        function about the 'y' axis at x == 0.
    """
    return lambda x: np.maximum(1. - np.exp(-x_scale * x), 0)


def StepFunc():
    """Returns a step mapping function.

    Returns a step mapping function for ease of use when creating
    output mappings. The returned function takes in a non-optional 'x'
    parameter.
    """
    return lambda x: float(x > 0.)


def ThresholdingArray(
        n_neurons, n_ensembles, threshold, intercept_width=0.15,
        function=LinearFunc(), radius=1., **kwargs):
    if not is_iterable(threshold):
        threshold = threshold * np.ones((n_ensembles, 1))
    else:
        threshold = np.atleast_2d(threshold)

    with nengo.Network(**kwargs) as net:
        with nengo.presets.ThresholdingEnsembles(
                0., intercept_width, radius=radius):
            net.thresholding = nengo.networks.EnsembleArray(
                n_neurons, n_ensembles)

        net.bias = nengo.Node(1.)
        nengo.Connection(net.bias, net.thresholding.input,
                         transform=-threshold)

        net.input = net.thresholding.input
        net.thresholded = net.thresholding.output
        net.output = net.thresholding.add_output('function', function)
    return net


def WTA(n_neurons, n_ensembles, inhibit_scale=1.0, inhibit_synapse=0.005,
        **kwargs):
    net = ThresholdingArray(n_neurons, n_ensembles, **kwargs)
    with net:
        nengo.Connection(
            net.thresholded, net.input,
            transform=inhibit_scale * (np.eye(n_ensembles) - 1.),
            synapse=inhibit_synapse)
    return net
