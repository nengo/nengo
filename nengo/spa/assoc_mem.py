import nengo
from nengo.spa.module import Module


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
                 input_keys=None, output_keys=None,
                 default_output_key=None, threshold=0.3, input_scale=1.0,
                 inhibitable=False, inhibit_scale=1.5, wta_output=False,
                 wta_inhibit_scale=3.0, wta_synapse=0.005,
                 threshold_output=False, label=None, seed=None,
                 add_to_container=None, **ens_args):
        super(AssociativeMemory, self).__init__(label, seed, add_to_container)

        if input_keys is None:
            input_keys = input_vocab.keys
            input_vectors = input_vocab.vectors
        else:
            input_vectors = input_vocab.create_subset(input_keys).vectors

        # If output vocabulary is not specified, use input vocabulary
        # (i.e autoassociative memory)
        if output_vocab is None:
            output_vocab = input_vocab
            output_vectors = input_vectors
            output_keys = input_keys
        else:
            if output_keys is None:
                output_keys = input_keys
            output_vectors = output_vocab.create_subset(output_keys).vectors

        if default_output_key is None:
            default_output_vector = None
        else:
            default_output_vector = output_vocab.parse(default_output_key).v

        # Create nengo network
        with self:
            self.am = nengo.networks.AssociativeMemory(
                input_vectors=input_vectors,
                output_vectors=output_vectors,
                default_output_vector=default_output_vector,
                threshold=threshold, input_scale=input_scale,
                inhibitable=inhibitable,
                inhibit_scale=inhibit_scale,
                wta_output=wta_output,
                wta_inhibit_scale=wta_inhibit_scale,
                wta_synapse=wta_synapse,
                threshold_output=threshold_output,
                label=label, seed=seed,
                add_to_container=add_to_container,
                **ens_args)

            for ens, inkey, outkey in zip(
                    self.am.am_ensembles, input_keys, output_keys):
                if inkey != outkey:
                    ens.label = "%s->%s" % (inkey, outkey)
                else:
                    ens.label = str(inkey)

            self.input = self.am.input
            self.output = self.am.output

            if inhibitable:
                self.inhibit = self.am.inhibit

            self.utilities = self.am.utilities
            if threshold_output:
                self.thresholded_utilities = self.am.thresholded_utilities

        self.inputs = dict(default=(self.input, input_vocab))
        self.outputs = dict(default=(self.output, output_vocab))
