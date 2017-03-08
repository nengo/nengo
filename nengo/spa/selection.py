"""Selection networks."""

import nengo
import numpy as np
from nengo.utils.compat import is_iterable


def ThresholdingArray(
        n_neurons, n_ensembles, threshold, intercept_width=0.15, function=None,
        radius=1., **kwargs):
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
        if function is None:
            net.output = net.thresholding.output
        else:
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
