"""Selection networks."""

import nengo
import numpy as np


def IA(
        n_neurons, n_ensembles, accum_threshold=0.8, accum_neuron_ratio=0.7,
        accum_timescale=0.1, feedback_timescale=0.005,
        accum_synapse=0.1, ff_synapse=0.005,
        intercept_width=0.15, radius=1., **kwargs):
    n_accum_neurons = int(accum_neuron_ratio * n_neurons)
    n_thresholding_neurons = n_neurons - n_accum_neurons

    bar_beta = 1. + radius * feedback_timescale / accum_timescale
    feedback_tr = (
        np.eye(n_ensembles) - bar_beta * (1. - np.eye(n_ensembles)) /
        feedback_timescale)

    with nengo.Network(**kwargs) as net:
        net.accumulators = ThresholdingArray(
            n_accum_neurons, n_ensembles, threshold=0., radius=radius)
        net.thresholding = ThresholdingArray(
            n_thresholding_neurons, n_ensembles, threshold=accum_threshold,
            radius=radius, function=lambda x: x > accum_threshold)

        nengo.Connection(
            net.accumulators.output, net.accumulators.input,
            synapse=accum_synapse)
        nengo.Connection(
            net.accumulators.output, net.thresholding.input,
            synapse=ff_synapse)
        nengo.Connection(
            net.thresholding.output, net.accumulators.input,
            synapse=accum_synapse, transform=accum_synapse * feedback_tr)

        net.input_reset = nengo.Node(size_in=1)
        nengo.Connection(
            net.input_reset, net.accumulators.input, synapse=None,
            transform=-radius * np.ones((n_ensembles, 1)) / accum_synapse)

        net.input = nengo.Node(size_in=n_ensembles)
        nengo.Connection(net.input, net.accumulators.input, synapse=None,
                         transform=1. / accum_timescale)
        net.output = net.thresholding.output
    return net


def ThresholdingArray(
        n_neurons, n_ensembles, threshold, intercept_width=0.15, function=None,
        radius=1., **kwargs):
    with nengo.Network(**kwargs) as net:
        with nengo.presets.ThresholdingEnsembles(
                0., intercept_width, radius=radius):
            net.thresholding = nengo.networks.EnsembleArray(
                n_neurons, n_ensembles)

        net.bias = nengo.Node(1.)
        nengo.Connection(net.bias, net.thresholding.input,
                         transform=-threshold * np.ones((n_ensembles, 1)))

        net.input = net.thresholding.input
        net.thresholded = net.thresholding.output

        if function is None:
            function = lambda x: x
        function = lambda x, function=function: function(x + threshold)
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
