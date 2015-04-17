import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.dists import Choice, Uniform
from nengo.utils.compat import is_iterable
from nengo.utils.network import with_self


class AssociativeMemory(nengo.Network):
    """Associative memory network.

    Parameters
    ----------
    input_vectors: list or numpy.array
    output_vectors: list of numpy.array, optional
        The list of vectors to be produced for each match. If
        not given, the associative memory will act like an auto-associative
        memory (cleanup memory).
    default_output_vector: numpy.array, optional
        The vector to be produced if the input value matches none of vectors
        in the input vector list.

    n_neurons_per_ensemble: int
        Number of neurons to use per ensemble in the associative memory

    threshold: float, list, optional
        The association activation threshold.
    input_scale: float, list, optional
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

    threshold_output: boolean, optional
        Flag to indicate if the output vector should be thresholded

    """
    def __init__(self, input_vectors, output_vectors=None,  # noqa: C901
                 default_output_vector=None, threshold=0.3, input_scale=1.0,
                 inhibitable=False, inhibit_scale=1.5, wta_output=False,
                 wta_inhibit_scale=3.0, wta_synapse=0.005,
                 threshold_output=False, label=None, seed=None,
                 add_to_container=None, **ens_args):
        super(AssociativeMemory, self).__init__(label, seed, add_to_container)

        label_prefix = "" if label is None else label + "_"
        n_neurons_per_ensemble = ens_args.get('n_neurons', 50)

        # If output vocabulary is not specified, use input vocabulary
        # (i.e autoassociative memory)
        if output_vectors is None:
            output_vectors = input_vectors

        # Handle different vocabulary types
        if is_iterable(input_vectors):
            input_vectors = np.matrix(input_vectors)

        if is_iterable(output_vectors):
            output_vectors = np.matrix(output_vectors)

        # Fail if number of input items and number of output items don't
        # match
        if input_vectors.shape[0] != output_vectors.shape[0]:
            raise ValueError(
                'Number of input vectors does not match number of output '
                'vectors. %d != %d'
                % (input_vectors.shape[0], output_vectors.shape[0]))

        # Handle possible different threshold / input_scale values for each
        # element in the associative memory
        if not is_iterable(threshold):
            threshold = np.array([threshold] * input_vectors.shape[0])
        else:
            threshold = np.array(threshold)
        if threshold.shape[0] != input_vectors.shape[0]:
            raise ValueError(
                'Number of threshold values do not match number of input'
                'vectors. Got: %d, expected %d.' %
                (threshold.shape[0], input_vectors.shape[0]))

        if not is_iterable(input_scale):
            input_scale = np.matrix([input_scale] * input_vectors.shape[0])
        else:
            input_scale = np.matrix(input_scale)
        if input_scale.shape[1] != input_vectors.shape[0]:
            raise ValueError(
                'Number of input_scale values do not match number of input'
                'vectors. Got: %d, expected %d.' %
                (input_scale.shape[1], input_vectors.shape[0]))

        # Input and output nodes
        N = input_vectors.shape[0]
        self.num_items = N

        with self:
            bias_node = nengo.Node(output=1)

            self.input = nengo.Node(size_in=input_vectors.shape[1],
                                    label="input")
            self.output = nengo.Node(size_in=output_vectors.shape[1],
                                     label="output")

            self.elem_input = nengo.Node(size_in=N, label="element input")
            self.elem_output = nengo.Node(size_in=N, label="element output")

            self.threshold_output = threshold_output

            nengo.Connection(self.input, self.elem_input, synapse=None,
                             transform=np.multiply(input_vectors,
                                                   input_scale.T))

            # Evaluation points parameters
            n_eval_points = 5000

            # Make each ensemble
            self.am_ensembles = []
            for i in range(N):
                # Ensemble array parameters
                ens_params = dict(ens_args)
                ens_params['radius'] = ens_args.get('radius', 1.0)
                ens_params['dimensions'] = 1
                ens_params['n_neurons'] = n_neurons_per_ensemble
                ens_params['intercepts'] = Uniform(threshold[i], 1)
                ens_params['encoders'] = Choice([[1]])
                ens_params['eval_points'] = Uniform(threshold[i], 1.2)
                ens_params['n_eval_points'] = n_eval_points
                ens_params['label'] = label_prefix + str(i)

                # Create ensemble
                e = nengo.Ensemble(**ens_params)
                self.am_ensembles.append(e)

                # Connect input and output nodes
                nengo.Connection(self.elem_input[i], e, synapse=None)
                nengo.Connection(e, self.elem_output[i], synapse=None)

            # Configure associative memory to be inhibitable
            if inhibitable:
                # Input node for inhibitory gating signal (if enabled)
                self.inhibit = nengo.Node(size_in=1, label="inhibit")
                nengo.Connection(self.inhibit, self.elem_input,
                                 transform=-np.ones((N, 1)) * inhibit_scale,
                                 synapse=None)
                # Note: We can use decoded connection here because all the
                # encoding vectors are [1]

            # Configure associative memory to have mutually inhibited output
            if wta_output:
                nengo.Connection(self.elem_output, self.elem_input,
                                 synapse=wta_synapse,
                                 transform=(np.eye(N) - 1) * wta_inhibit_scale)

            # Configure utilities output
            self.utilities = self.elem_output

            # Configure default output vector
            if default_output_vector is not None or threshold_output:
                default_threshold = min(1 - np.min(threshold), 0.9)

                ens_params = dict(ens_args)
                ens_params['radius'] = ens_args.get('radius', 1.0)
                ens_params['dimensions'] = 1
                ens_params['n_neurons'] = n_neurons_per_ensemble
                ens_params['intercepts'] = Uniform(default_threshold, 1)
                ens_params['encoders'] = Choice([[1]])
                ens_params['eval_points'] = Uniform(default_threshold, 1.1)
                ens_params['n_eval_points'] = n_eval_points
                ens_params['label'] = "default vector gate"

                default_vector_gate = nengo.Ensemble(**ens_params)

                nengo.Connection(bias_node, default_vector_gate, synapse=None)
                nengo.Connection(self.elem_output, default_vector_gate,
                                 transform=-2 * np.ones((1, N)), synapse=0.01)

                self.default_output_utility = default_vector_gate
                self.default_output_thresholded_utility = default_vector_gate

                if default_output_vector is not None:
                    nengo.Connection(
                        default_vector_gate, self.output,
                        transform=np.matrix(default_output_vector).T,
                        synapse=None)

                if inhibitable:
                    nengo.Connection(self.inhibit, default_vector_gate,
                                     transform=[[-1]], synapse=None)

            # Set up thresholding ensembles
            if threshold_output:
                # Ensemble array parameters
                ens_params = dict(ens_args)
                ens_params['radius'] = ens_args.get('radius', 1.0)
                ens_params['n_neurons'] = n_neurons_per_ensemble
                ens_params['n_ensembles'] = N
                ens_params['intercepts'] = Uniform(0.5, 1)
                ens_params['encoders'] = Choice([[1]])
                ens_params['eval_points'] = Uniform(0.5, 1.1)
                ens_params['n_eval_points'] = n_eval_points

                self.thresh_ens = EnsembleArray(**ens_params)
                self.thresholded_utilities = self.thresh_ens.output

                nengo.Connection(bias_node, self.thresh_ens.input,
                                 transform=np.ones((N, 1)), synapse=None)
                if wta_output:
                    nengo.Connection(self.elem_output, self.thresh_ens.input,
                                     transform=10 * (np.eye(N) - 1),
                                     synapse=0.01)
                else:
                    ens_params['intercepts'] = Uniform(0.25, 1)
                    self.thresh_ens_int = EnsembleArray(**ens_params)
                    nengo.Connection(bias_node, self.thresh_ens_int.input,
                                     transform=np.ones((N, 1)), synapse=None)
                    nengo.Connection(self.elem_output,
                                     self.thresh_ens_int.input,
                                     transform=-10, synapse=0.005)
                    nengo.Connection(self.thresh_ens_int.output,
                                     self.thresh_ens.input,
                                     transform=-10, synapse=0.005)

                nengo.Connection(self.thresh_ens.output, self.output,
                                 transform=output_vectors.T, synapse=None)

                nengo.Connection(default_vector_gate, self.thresh_ens.input,
                                 transform=-2 * np.ones((N, 1)), synapse=0.01)

                if inhibitable:
                    nengo.Connection(
                        self.inhibit, self.thresh_ens.input,
                        transform=-np.ones((N, 1)) * inhibit_scale,
                        synapse=None)
            else:
                nengo.Connection(self.elem_output, self.output,
                                 transform=output_vectors.T, synapse=None)

    @with_self
    def add_input(self, name, input_vectors, input_scale=1.0):
        # Handle different vocabulary types
        if is_iterable(input_vectors):
            input_vectors = np.matrix(input_vectors)

        # Handle possible different input_scale values for each
        # element in the associative memory
        if not is_iterable(input_scale):
            input_scale = np.matrix([input_scale] * input_vectors.shape[0])
        else:
            input_scale = np.matrix(input_scale)
        if input_scale.shape[1] != input_vectors.shape[0]:
            raise ValueError(
                'Number of input_scale values do not match number of input '
                'vectors. Got: %d, expected %d.' %
                (input_scale.shape[1], input_vectors.shape[0]))

        input = nengo.Node(size_in=input_vectors.shape[1], label=name)

        if hasattr(self, name):
            raise NameError('Name "%s" already exists as a node in the'
                            'associative memory.')
        else:
            setattr(self, name, input)

        nengo.Connection(input, self.elem_input,
                         synapse=None,
                         transform=np.multiply(input_vectors, input_scale.T))

    @with_self
    def add_output(self, name, output_vectors):
        # Handle different vocabulary types
        if is_iterable(output_vectors):
            output_vectors = np.matrix(output_vectors)

        output = nengo.Node(size_in=output_vectors.shape[1], label=name)

        if hasattr(self, name):
            raise NameError('Name "%s" already exists as a node in the'
                            'associative memory.')
        else:
            setattr(self, name, output)

        if self.threshold_output:
            nengo.Connection(self.thresh_ens.output, output, synapse=None,
                             transform=output_vectors.T)
        else:
            nengo.Connection(self.elem_output, output, synapse=None,
                             transform=output_vectors.T)
