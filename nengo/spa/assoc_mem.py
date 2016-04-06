import numpy as np

import nengo
from nengo.dists import Choice, Uniform
from nengo.networks.ensemblearray import EnsembleArray
from nengo.spa.module import Module
from nengo.spa.vocab import Vocabulary
from nengo.utils.compat import is_iterable


class AssociativeMemory(Module):
    """Associative memory module.

    Parameters
    ----------
    input_vocab: list of numpy.array, spa.Vocabulary
        The vocabulary (or list of vectors) to match.
    output_vocab: list of numpy.array, spa.Vocabulary, optional
        The vocabulary (or list of vectors) to be produced for each match. If
        not given, the associative memory will act like an auto-associative
        memory (cleanup memory).
    default_output_vector: numpy.array, spa.SemanticPointer, optional
        The vector to be produced if the input value matches none of vectors
        in the input vector list.
    threshold: float, optional
        The association activation threshold.
    input_scale: float, optional
        Scaling factor to apply on the input vectors.

    inhibitable: boolean, optional
        Flag to indicate if the entire associative memory module is
        inhibitable (entire thing can be shut off).
    inhibit_scale: float, optional
        Scaling factor on the gating connections (must have inhibitable =
        True). Setting a larger value will ensure that the cleanup memory
        output is inhibited at a faster rate, however, recovery of the
        network when inhibition is released will be slower.

    wta_output: boolean, optional
        Flag to indicate if output of the associative memory should contain
        more than one vectors. Set to True if only one vectors output is
        desired -- i.e. a winner-take-all (wta) output. Leave as default
        (False) if (possible) combinations of vectors is desired.
    wta_inhibit_scale: float, optional
        Scaling factor on the winner-take-all (wta) inhibitory connections.
    wta_synapse: float, optional
        Synapse to use for the winner-take-all (wta) inhibitory connections.

    output_utilities: boolean, optional
        Flag to indicate if the direct utilities (in addition to the vectors)
        are output as well.
    output_thresholded_utilities: boolean, optional
        Flag to indicate if the direct thresholded utilities (in addition to
        the vectors) are output as well.

    neuron_type: nengo.Neurons, optional
        Neuron type to use in the associative memory. Defaults to
    n_neurons_per_ensemble: int, optional
        Number of neurons per ensemble in the associative memory. There is
        one ensemble created per vector being compared.

    """

    def __init__(self, input_vocab, output_vocab=None,  # noqa: C901
                 default_output_vector=None, threshold=0.3, input_scale=1.0,
                 inhibitable=False, inhibit_scale=1.0, wta_output=False,
                 wta_inhibit_scale=2.0, wta_synapse=0.005,
                 output_utilities=False, output_thresholded_utilities=False,
                 neuron_type=nengo.LIF(), n_neurons_per_ensemble=10,
                 label=None, seed=None, add_to_container=None):
        super(AssociativeMemory, self).__init__(label, seed, add_to_container)

        # If output vocabulary is not specified, use input vocabulary
        # (i.e autoassociative memory)
        if output_vocab is None:
            output_vocab = input_vocab

        # Handle different vocabulary types
        if isinstance(input_vocab, Vocabulary):
            input_vectors = input_vocab.vectors
        elif is_iterable(input_vocab):
            input_vectors = np.matrix(input_vocab)
        else:
            input_vectors = input_vocab

        if isinstance(output_vocab, Vocabulary):
            output_vectors = output_vocab.vectors
        elif is_iterable(output_vocab):
            output_vectors = np.matrix(output_vocab)
        else:
            output_vectors = output_vocab

        # Fail if number of input items and number of output items don't match
        if input_vectors.shape[0] != output_vectors.shape[0]:
            raise ValueError(
                'number of input vectors does not match number of output '
                'vectors. %d != %d'
                % (input_vectors.shape[0], output_vectors.shape[0]))

        N = len(input_vectors)
        n_eval_points = 500
        eval_point_margin = 0.1
        eval_points = Uniform(
            threshold + eval_point_margin, 1 + eval_point_margin)

        # Ensemble array parameters
        ea_params = {'radius': 1.0,
                     'neuron_type': neuron_type,
                     'n_neurons': n_neurons_per_ensemble,
                     'n_ensembles': N,
                     'intercepts': Uniform(threshold, 1),
                     'max_rates': Uniform(100, 200),
                     'encoders': Choice([[1]]),
                     'n_eval_points': n_eval_points,
                     'eval_points': eval_points}

        # Thresholding function
        def threshold_func(x):
            return x > threshold

        with self:
            # Input and output nodes
            self.input = nengo.Node(size_in=input_vectors.shape[1],
                                    label="input")
            self.output = nengo.Node(size_in=output_vectors.shape[1],
                                     label="output")

            # Ensemble array to do the thresholding stuff
            self.thresholded_ens_array = EnsembleArray(
                label="thresholded ens array", **ea_params)

            # Connect input and output nodes
            nengo.Connection(self.input, self.thresholded_ens_array.input,
                             synapse=None,
                             transform=input_vectors * input_scale)
            nengo.Connection(
                self.thresholded_ens_array.add_output(
                    'thresholded_output', threshold_func),
                self.output, synapse=None, transform=output_vectors.T)

            # Configure associative memory to be inhibitable
            if inhibitable:
                # Input node for inhibitory gating signal (if enabled)
                self.inhibit = nengo.Node(size_in=1, label="inhibit")
                nengo.Connection(
                    self.inhibit, self.thresholded_ens_array.input,
                    synapse=None, transform=-np.ones((N, 1)))
                # Note: We can use decoded connection here because all the
                # encoding vectors are [1]

            # Configure associative memory to have mutually inhibited output
            if wta_output:
                nengo.Connection(self.thresholded_ens_array.output,
                                 self.thresholded_ens_array.input,
                                 synapse=wta_synapse,
                                 transform=(np.eye(N) - 1) * inhibit_scale)

            # Configure utilities output
            if output_utilities:
                self.utilities = nengo.Node(size_in=N, label="utilities")
                nengo.Connection(self.thresholded_ens_array.output,
                                 self.utilities, synapse=None)

            # Configure utilities output
            if output_thresholded_utilities:
                self.thresholded_utilities = nengo.Node(
                    size_in=N, label="thresholded_utilities")
                nengo.Connection(self.thresholded_ens_array.thresholded_output,
                                 self.thresholded_utilities, synapse=None)

            # Configure default output vector
            if default_output_vector is not None:
                eval_points = Uniform(0.8, 1)
                bias = nengo.Node(output=[1])
                default_vector_gate = nengo.Ensemble(
                    n_neurons_per_ensemble, dimensions=1,
                    encoders=Choice([[1]]),
                    intercepts=Uniform(0.5, 1),
                    max_rates=ea_params['max_rates'],
                    n_eval_points=n_eval_points,
                    eval_points=eval_points,
                    label="default vector gate")
                nengo.Connection(bias, default_vector_gate, synapse=None)
                nengo.Connection(self.thresholded_ens_array.thresholded_output,
                                 default_vector_gate,
                                 transform=-np.ones((1, N)),
                                 synapse=0.005)
                nengo.Connection(default_vector_gate, self.output,
                                 transform=np.matrix(default_output_vector).T,
                                 synapse=None)
                if inhibitable:
                    nengo.Connection(self.inhibit, default_vector_gate,
                                     synapse=None, transform=[[-1]])

        if isinstance(input_vocab, Vocabulary):
            self.inputs = dict(default=(self.input, input_vocab))
        if isinstance(output_vocab, Vocabulary):
            self.outputs = dict(default=(self.output, output_vocab))
