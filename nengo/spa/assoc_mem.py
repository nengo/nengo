from nengo.networks.assoc_mem import AssociativeMemory as AssocMem
from nengo.spa.module import Module


class AssociativeMemory(Module):
    """Associative memory module.

    See :doc:`examples/spa/associative-memory` for an introduction and
    examples.

    Parameters
    ----------
    input_vocab: Vocabulary
        The vocabulary to match.
    output_vocab: Vocabulary, optional
        The vocabulary to be produced for each match. If
        None, the associative memory will act like an autoassociative memory
        (cleanup memory).
    input_keys : list, optional
        A list of strings that correspond to the input vectors.
    output_keys : list, optional
        A list of strings that correspond to the output vectors.
    default_output_key: str, optional
        The semantic pointer string to be produced if the input value matches
        none of vectors in the input vector list.
    threshold: float, optional
        The association activation threshold.
    inhibitable: bool, optional
        Flag to indicate if the entire associative memory module is
        inhibitable (i.e., the entire module can be inhibited).
    wta_output: bool, optional
        Flag to indicate if output of the associative memory should contain
        more than one vector. If True, only one vector's output will be
        produced; i.e. produce a winner-take-all (WTA) output.
        If False, combinations of vectors will be produced.
    wta_inhibit_scale: float, optional
        Scaling factor on the winner-take-all (WTA) inhibitory connections.
    wta_synapse: float, optional
        Synapse to use for the winner-take-all (wta) inhibitory connections.
    threshold_output: bool, optional
        Adds a threholded output if True.
    label : str, optional
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional
        The seed used for random number generation.
    add_to_container : bool, optional
        Determines if this Network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    def __init__(
        self,
        input_vocab,
        output_vocab=None,  # noqa: C901
        input_keys=None,
        output_keys=None,
        default_output_key=None,
        threshold=0.3,
        inhibitable=False,
        wta_output=False,
        wta_inhibit_scale=3.0,
        wta_synapse=0.005,
        threshold_output=False,
        label=None,
        seed=None,
        add_to_container=None,
    ):
        super().__init__(label, seed, add_to_container)

        if input_keys is None:
            input_keys = input_vocab.keys
            input_vectors = input_vocab.vectors
        else:
            input_vectors = [input_vocab.parse(key).v for key in input_keys]

        # If output vocabulary is not specified, use input vocabulary
        # (i.e autoassociative memory)
        if output_vocab is None:
            output_vocab = input_vocab
            output_vectors = input_vectors
        else:
            if output_keys is None:
                output_keys = input_keys

            output_vectors = [output_vocab.parse(key).v for key in output_keys]

        if default_output_key is None:
            default_output_vector = None
        else:
            default_output_vector = output_vocab.parse(default_output_key).v

        # Create nengo network
        with self:
            self.am = AssocMem(
                input_vectors=input_vectors,
                output_vectors=output_vectors,
                threshold=threshold,
                inhibitable=inhibitable,
                label=label,
                seed=seed,
                add_to_container=add_to_container,
            )

            if default_output_vector is not None:
                self.am.add_default_output_vector(default_output_vector)

            if wta_output:
                self.am.add_wta_network(wta_inhibit_scale, wta_synapse)

            if threshold_output:
                self.am.add_threshold_to_outputs()

            self.input = self.am.input
            self.output = self.am.output

            if inhibitable:
                self.inhibit = self.am.inhibit

            self.utilities = self.am.utilities
            if threshold_output:
                self.thresholded_utilities = self.am.thresholded_utilities

        self.inputs = dict(default=(self.input, input_vocab))
        self.outputs = dict(default=(self.output, output_vocab))
