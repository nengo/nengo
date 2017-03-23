"""Associative memory implementations.

See :doc:`examples/associative_memory` for an introduction and examples.
"""

import nengo
import numpy as np
from nengo.spa.module import Module
from nengo.spa.selection import IA, ThresholdingArray, WTA
from nengo.spa.vocab import VocabularyOrDimParam
from nengo.utils.network import with_self


class AssociativeMemory(Module):
    """General associative memory module.

    This provides a low-level selection network with the necessary interface
    to include it within the SPA system.

    Parameters
    ----------
    selection_net : Network
        The network that is used to select the response. It needs to accept
        the arguments *n_neurons* (number of neurons to use to represent each
        possible choice) and *n_ensembles* (number of choices). The returned
        network needs to have an *input* attribute to which the utilities for
        each choice are connected and an *output* attribute from which a
        connection will be created to read the selected output(s).
    input_vocab: list or Vocabulary
        The vocabulary (or list of vectors) to match.
    output_vocab: list or Vocabulary, optional (Default: None)
        The vocabulary (or list of vectors) to be produced for each match. If
        None, the associative memory will act like an autoassociative memory
        (cleanup memory).
    input_keys : list, optional (Default: None)
        A list of strings that correspond to the input vectors.
    output_keys : list, optional (Default: None)
        A list of strings that correspond to the output vectors.
    n_neurons : int
        Number of neurons to represent each choice, passed on to the
        *selection_net*.
    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if this Network will be added to the current container.
        If None, will be true if currently with
    vocabs : VocabularyMap, optional (Default: None)
        Maps dimensionalities to the corresponding default vocabularies.
    """
    input_vocab = VocabularyOrDimParam(
        'input_vocab', default=None, readonly=True)
    output_vocab = VocabularyOrDimParam(
        'output_vocab', default=None, readonly=True)

    def __init__(
            self, selection_net, input_vocab, output_vocab=None,
            input_keys=None, output_keys=None, n_neurons=50,
            label=None, seed=None, add_to_container=None, vocabs=None,
            **selection_net_args):
        super(AssociativeMemory, self).__init__(
            label=label, seed=seed, add_to_container=add_to_container,
            vocabs=vocabs)

        if output_vocab is None:
            output_vocab = input_vocab
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        if input_keys is None:
            input_keys = self.input_vocab.keys()
            input_vectors = self.input_vocab.vectors
        else:
            input_vectors = [input_vocab.parse(key).v for key in input_keys]

        if output_keys is None:
            output_keys = input_keys
        output_vectors = [output_vocab.parse(key).v for key in output_keys]

        input_vectors = np.asarray(input_vectors)
        output_vectors = np.asarray(output_vectors)

        with self:
            self.selection = selection_net(
                n_neurons=n_neurons, n_ensembles=len(input_vectors),
                label="selection", **selection_net_args)
            self.input = nengo.Node(size_in=self.input_vocab.dimensions,
                                    label="input")
            self.output = nengo.Node(size_in=self.output_vocab.dimensions,
                                     label="output")

            nengo.Connection(
                self.input, self.selection.input, transform=input_vectors)
            nengo.Connection(
                self.selection.output, self.output, transform=output_vectors.T)

        self.inputs = dict(default=(self.input, self.input_vocab))
        self.outputs = dict(default=(self.output, self.output_vocab))

    @with_self
    def add_default_output(self, key, min_activation_value, n_neurons=50):
        """Adds a Semantic Pointer to output when no other pointer is active.

        Parameters
        ----------
        key : str
            Semantic Pointer to output.
        min_activation_value : float
            Minimum output of another Semantic Pointer to deactivate the
            default output.
        n_neurons : int, optional (Default: 50)
            Number of neurons used to represent the default Semantic Pointer.
        """
        assert not hasattr(self, 'default_ens'), \
            "Can add default output only once."

        with nengo.presets.ThresholdingEnsembles(0.):
            setattr(self, 'default_ens',
                    nengo.Ensemble(n_neurons, 1, label="default"))
        setattr(self, 'bias', nengo.Node(1., label="bias"))
        nengo.Connection(self.bias, self.default_ens)
        nengo.Connection(
            self.default_ens, self.output,
            transform=np.atleast_2d(self.output_vocab.parse(key).v).T)
        nengo.Connection(
            self.selection.output, self.default_ens,
            transform=-np.ones(
                (1, self.selection.output.size_out)) / min_activation_value)


class IaAssocMem(AssociativeMemory):
    """Associative memory based on the `IA` network.

    See `AssociativeMemory` and `IA` for more information.
    """
    def __init__(
            self, input_vocab, output_vocab=None, input_keys=None,
            output_keys=None, n_neurons=50, label=None, seed=None,
            add_to_container=None, vocabs=None, **selection_net_args):
        super(IaAssocMem, self).__init__(
            selection_net=IA,
            input_vocab=input_vocab, output_vocab=output_vocab,
            input_keys=input_keys, output_keys=output_keys,
            n_neurons=n_neurons, label=label, seed=seed,
            add_to_container=add_to_container, vocabs=vocabs,
            **selection_net_args)
        self.input_reset = self.selection.input_reset
        self.inputs['reset'] = (self.input_reset, None)


class ThresholdingAssocMem(AssociativeMemory):
    """Associative memory based on `ThresholdingArray`.

    See `AssociativeMemory` and `ThresholdingArray` for more information.
    """
    def __init__(
            self, threshold, input_vocab, output_vocab=None, input_keys=None,
            output_keys=None, n_neurons=50, label=None, seed=None,
            add_to_container=None, vocabs=None, **selection_net_args):
        selection_net_args['threshold'] = threshold
        super(ThresholdingAssocMem, self).__init__(
            selection_net=ThresholdingArray,
            input_vocab=input_vocab, output_vocab=output_vocab,
            input_keys=input_keys, output_keys=output_keys,
            n_neurons=n_neurons, label=label, seed=seed,
            add_to_container=add_to_container, vocabs=vocabs,
            **selection_net_args)


class WtaAssocMem(AssociativeMemory):
    """Associative memory based on the `WTA` network.

    See `AssociativeMemory` and `WTA` for more information.
    """
    def __init__(
            self, threshold, input_vocab, output_vocab=None, input_keys=None,
            output_keys=None, n_neurons=50, label=None, seed=None,
            add_to_container=None, vocabs=None, **selection_net_args):
        selection_net_args['threshold'] = threshold
        super(WtaAssocMem, self).__init__(
            selection_net=WTA,
            input_vocab=input_vocab, output_vocab=output_vocab,
            input_keys=input_keys, output_keys=output_keys,
            n_neurons=n_neurons, label=label, seed=seed,
            add_to_container=add_to_container, vocabs=vocabs,
            **selection_net_args)
