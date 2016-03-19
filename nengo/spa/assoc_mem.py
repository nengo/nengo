from nengo.networks.assoc_mem import AssociativeMemory as AssocMem
from nengo.spa.module import Module


class AssociativeMemory(Module):
    """Associative memory module.

    See :doc:`examples/associative_memory` for an introduction and examples.

    Parameters
    ----------
    input_vocab: list or Vocabulary
        The vocabulary (or list of vectors) to match.
    output_vocab: list or Vocabulary, optional (Default: None)
        The vocabulary (or list of vectors) to be produced for each match. If
        not given, the associative memory will act like an auto-associative
        memory (cleanup memory).

    input_keys: list of strings, optional (Default: None)
        List of keys (ordered) from the input vocabulary to use as the input
        semantic pointers for the associative memory.
    output_keys: list of strings, optional (Default: None)
        List of keys (ordered) from the output vocabulary to use as the output
        semantic pointers for the associative memory.

    default_output_vector: numpy.array, spa.SemanticPointer, optional (Default: None)
        The vector to be produced if the input value matches none of vectors
        in the input vector list.
    threshold: float, optional (Default: 0.3)
        The association activation threshold.

    inhibitable: boolean, optional
        Flag to indicate if the entire associative memory module is
        inhibitable (entire thing can be shut off).

    wta_output: boolean, optional (Default: False)
        Flag to indicate if output of the associative memory should contain
        more than one vector. If True, only one vector's output will be
        produced; i.e. produce a winner-take-all (WTA) output.
        If False, combinations of vectors will be produced.
    wta_inhibit_scale: float, optional (Default: 3.0)
        Scaling factor on the winner-take-all (WTA) inhibitory connections.
    wta_synapse: float, optional (Default: 0.005)
        Synapse to use for the winner-take-all (wta) inhibitory connections.

    cleanup_output: boolean, optional (Default: False)
        Create the associative memory with cleaned outputs as well as the
        standard outputs.
    replace_output_with_cleaned_output: boolean, optional
        Set to true to use the cleaned outputs as the default output of the
        associative memory module.
    """

    def __init__(self, input_vocab, output_vocab=None,  # noqa: C901
                 input_keys=None, output_keys=None,
                 default_output_key=None, threshold=0.3,
                 inhibitable=False, wta_output=False,
                 wta_inhibit_scale=3.0, wta_synapse=0.005,
                 cleanup_output=False,
                 replace_output_with_cleaned_output=False,
                 label=None, seed=None, add_to_container=None):
        super(AssociativeMemory, self).__init__(label, seed, add_to_container)

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
            self.am = AssocMem(input_vectors=input_vectors,
                               output_vectors=output_vectors,
                               threshold=threshold,
                               inhibitable=inhibitable,
                               label=label, seed=seed,
                               add_to_container=add_to_container)

            if default_output_vector is not None:
                self.am.add_default_output_vector(default_output_vector)

            if wta_output:
                self.am.add_wta_network(wta_inhibit_scale, wta_synapse)

            if cleanup_output:
                self.am.add_cleanup_output(
                    replace_output=replace_output_with_cleaned_output)

            self.input = self.am.input
            self.output = self.am.output

            if cleanup_output and not replace_output_with_cleaned_output:
                self.cleaned_output = self.am.cleaned_output

            if inhibitable:
                self.inhibit = self.am.inhibit

            self.utilities = self.am.output_utilities
            if cleanup_output:
                self.cleaned_utilities = self.am.cleaned_output_utilities

        self.inputs = dict(default=(self.input, input_vocab))
        self.outputs = dict(default=(self.output, output_vocab))
