"""Helper functions for constructing nengo models that implement actions."""
import nengo
import numpy as np


def convolution(module, target_name, effect, n_neurons_cconv, synapse):
    """Implement an action_objects.Convolution.

    Parameters
    ----------
    module : spa.Module
        The module that will own this convolution
    target_name : string
        The name of the object to send the convolution result to
    effect : action_objects.Convolution
        The details of the convolution to implement
    n_neurons_cconv : int
        Number of neurons in each product population
    synapse : float (or nengo.Synapse)
        The synapse to use for connections into and out of the convolution

    Returns the created nengo.networks.CircularConvolution.
    """
    source1 = effect.source1
    source2 = effect.source2

    target, target_vocab = module.spa.get_module_input(target_name)
    s1_output, s1_vocab = module.spa.get_module_output(source1.name)
    s2_output, s2_vocab = module.spa.get_module_output(source2.name)

    with module:
        cconv = nengo.networks.CircularConvolution(
            n_neurons_cconv, s1_vocab.dimensions,
            invert_a=False,
            invert_b=False,
            net=nengo.Network(label='cconv_%s' % str(effect)))

    with module.spa:
        # compute the requested transform
        t = s1_vocab.parse(str(effect.transform)).get_convolution_matrix()
        # handle conversion between different Vocabularies
        if target_vocab is not s1_vocab:
            t = np.dot(s1_vocab.transform_to(target_vocab), t)

        nengo.Connection(cconv.output, target, transform=t, synapse=synapse)

        t1 = s1_vocab.parse(source1.transform.symbol).get_convolution_matrix()
        if source1.inverted:
            D = s1_vocab.dimensions
            t1 = np.dot(t1, np.eye(D)[-np.arange(D)])

        nengo.Connection(s1_output, cconv.A, transform=t1, synapse=synapse)

        t2 = s2_vocab.parse(source2.transform.symbol).get_convolution_matrix()
        if source2.inverted:
            D = s2_vocab.dimensions
            t2 = np.dot(t2, np.eye(D)[-np.arange(D)])
        if s1_vocab is not s2_vocab:
            t2 = np.dot(s2_vocab.transform_to(s1_vocab), t2)
        nengo.Connection(s2_output, cconv.B, transform=t2, synapse=synapse)
    return cconv
