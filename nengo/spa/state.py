import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import Default, IntParam, NumberParam
from nengo.spa.module import Module
from nengo.spa.vocab import VocabularyOrDimParam


class State(Module):
    """A module capable of representing a single vector, with optional memory.

    This is a minimal SPA module, useful for passing data along (for example,
    visual input).

    Parameters
    ----------
    vocab : Vocabulary or int
        The vocabulary to use to interpret the vector. If an integer is given,
        the default vocabulary of that dimensionality will be used.
    subdimensions : int, optional (Default: 16)
        Size of the individual ensembles making up the vector.
        Must divide ``dimensions`` evenly.
    neurons_per_dimensions : int, optional (Default: 50)
        Number of neurons in an ensemble will be
        ``neurons_per_dimensions * subdimensions``.
    feedback : float, optional (Default: 0.0)
        Gain of feedback connection. Set to 1.0 for perfect memory,
        or 0.0 for no memory. Values in between will create a decaying memory.
    feedback_synapse : float, optional (Default: 0.1)
        The synapse on the feedback connection.
    kwargs
        Keyword arguments passed through to ``spa.Module``.
    """

    vocab = VocabularyOrDimParam('vocab', default=None, readonly=True)
    subdimensions = IntParam('subdimensions', default=16, low=1, readonly=True)
    neurons_per_dimension = IntParam(
        'neurons_per_dimension', default=50, low=1, readonly=True)
    feedback = NumberParam('feedback', default=.0, readonly=True)
    feedback_synapse = NumberParam(
        'feedback_synapse', default=.1, readonly=True)

    def __init__(self, vocab=Default, subdimensions=Default,
                 neurons_per_dimension=Default, feedback=Default,
                 feedback_synapse=Default, **kwargs):
        super(State, self).__init__(**kwargs)

        self.vocab = vocab
        self.subdimensions = subdimensions
        self.neurons_per_dimension = neurons_per_dimension
        self.feedback = feedback
        self.feedback_synapse = feedback_synapse

        dimensions = self.vocab.dimensions

        if dimensions % self.subdimensions != 0:
            raise ValidationError(
                "Dimensions (%d) must be divisible by subdimensions (%d)" % (
                    dimensions, self.subdimensions),
                attr='dimensions', obj=self)

        with self:
            self.state_ensembles = EnsembleArray(
                self.neurons_per_dimension * self.subdimensions,
                dimensions // self.subdimensions,
                ens_dimensions=self.subdimensions,
                radius=np.sqrt(float(self.subdimensions) / dimensions),
                #  TODO radius
                label='state')

            if self.feedback is not None and self.feedback != 0.0:
                nengo.Connection(
                    self.state_ensembles.output, self.state_ensembles.input,
                    transform=self.feedback, synapse=self.feedback_synapse)

        self.input = self.state_ensembles.input
        self.output = self.state_ensembles.output
        self.inputs = dict(default=(self.input, self.vocab))
        self.outputs = dict(default=(self.output, self.vocab))
