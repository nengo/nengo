import nengo
import numpy as np
from nengo.spa.module import Module
from nengo.spa.selection import IA, ThresholdingArray, WTA
from nengo.spa.vocab import VocabularyOrDimParam
from nengo.utils.network import with_self


class AssociativeMemory(Module):
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
