"""Helper functions for constructing nengo models that implement actions."""

import nengo
from nengo.spa.spa_ast import Artifact
from nengo.exceptions import SpaModuleError


def convolution(model, target_name, lhs, rhs, n_neurons_cconv, synapse):
    """Implement an action_objects.Convolution.

    Parameters
    ----------
    model : spa.Module
        The module that will own this convolution
    target_name : string
        The name of the object to send the convolution result to
    lhs : spa.spa_ast.Artifact
        Left hand side inputs.
    rhs : spa.spa_ast.Artifact
        Right hand side inputs.
    n_neurons_cconv : int
        Number of neurons in each product population
    synapse : float (or nengo.Synapse)
        The synapse to use for connections into and out of the convolution

    Returns the created nengo.networks.CircularConvolution.
    """

    try:
        target_vocab = model.get_module_input(target_name)[1]
    except SpaModuleError:
        target_vocab = model.get_module_input(target_name + '.default')[1]

    with model:
        cconv = nengo.networks.CircularConvolution(
            n_neurons_cconv, target_vocab.dimensions,
            invert_a=False, invert_b=False,
            net=nengo.Network(label='cconv'))

        for artifact in lhs:
            nengo.Connection(
                artifact.nengo_source, cconv.A, transform=artifact.transform,
                synapse=synapse)
        for artifact in rhs:
            nengo.Connection(
                artifact.nengo_source, cconv.B, transform=artifact.transform,
                synapse=synapse)
    return [Artifact(cconv.output)]


def product(model, target_name, lhs, rhs, n_neurons, synapse):
    try:
        target_vocab = model.get_module_input(target_name)[1]
    except SpaModuleError:
        target_vocab = model.get_module_input(target_name + '.default')[1]

    with model:
        prod = nengo.networks.Product(
            n_neurons, target_vocab.dimensions,
            net=nengo.Network(label='cconv'))

        for artifact in lhs:
            nengo.Connection(
                artifact.nengo_source, prod.A, transform=artifact.transform,
                synapse=synapse)
        for artifact in rhs:
            nengo.Connection(
                artifact.nengo_source, prod.B, transform=artifact.transform,
                synapse=synapse)
    return [Artifact(prod.output)]
