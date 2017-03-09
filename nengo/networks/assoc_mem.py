import warnings

import numpy as np

import nengo
from nengo.dists import Choice, Exponential, Uniform
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_iterable, range
from nengo.utils.network import with_self
from .ensemblearray import EnsembleArray


def filtered_step(t, shift=0.5, scale=50, step_val=1):
    return np.maximum(-1 / np.exp((t - shift) * scale) + 1, 0) * step_val


class AssociativeMemory(nengo.Network):
    """Associative memory network.

    Parameters
    ----------
    input_vectors: array_like
        The list of vectors to be compared against.
    output_vectors: array_like, optional (Default: None)
        The list of vectors to be produced for each match. If None, the
        associative memory will be autoassociative (cleanup memory).
    n_neurons: int, optional (Default: 50)
        The number of neurons for each of the ensemble (where each ensemble
        represents each item in the input_vectors list).
    threshold: float, optional (Default: 0.3)
        The association activation threshold.
    input_scales: float or array_like, optional (Default: 1.0)
        Scaling factor to apply on each of the input vectors. Note that it
        is possible to scale each vector independently.
    inhibitable: bool, optional (Default: False)
        Flag to indicate if the entire associative memory module is
        inhibitable (entire thing can be shut off). The input gain into
        the inhibitory connection is 1.5.
    label : str, optional (Default: None)
        A name for the ensemble. Used for debugging and visualization.
    seed : int, optional (Default: None)
        The seed used for random number generation.
    add_to_container : bool, optional (Default: None)
        Determines if the network will be added to the current container.
        If None, will be true if currently within a Network.
    """

    exp_scale = 0.15  # Scaling factor for exponential distribution
    n_eval_points = 5000

    def __init__(self, input_vectors, output_vectors=None,  # noqa: C901
                 n_neurons=50, threshold=0.3, input_scales=1.0,
                 inhibitable=False,
                 label=None, seed=None, add_to_container=None):
        super(AssociativeMemory, self).__init__(label, seed, add_to_container)

        # --- Put arguments in canonical form
        if output_vectors is None:
            # If output vocabulary is not specified, use input vector list
            # (i.e autoassociative memory)
            output_vectors = input_vectors
        if is_iterable(input_vectors):
            input_vectors = np.array(input_vectors, ndmin=2)
        if is_iterable(output_vectors):
            output_vectors = np.array(output_vectors, ndmin=2)

        if input_vectors.shape[0] == 0:
            raise ValidationError("Number of input vectors cannot be 0.",
                                  attr='input_vectors', obj=self)
        elif input_vectors.shape[0] != output_vectors.shape[0]:
            # Fail if number of input items and number of output items don't
            # match
            raise ValidationError(
                "Number of input vectors does not match number of output "
                "vectors. %d != %d"
                % (input_vectors.shape[0], output_vectors.shape[0]),
                attr='input_vectors', obj=type(self))

        # Handle possible different threshold / input_scale values for each
        # element in the associative memory
        if not is_iterable(threshold):
            threshold = threshold * np.ones(input_vectors.shape[0])
        else:
            threshold = np.array(threshold)

        # --- Check preconditions
        self.n_items = input_vectors.shape[0]
        if self.n_items != output_vectors.shape[0]:
            raise ValidationError(
                "Number of input vectors (%d) does not match number of output "
                "vectors (%d)" % (self.n_items, output_vectors.shape[0]),
                attr='input_vectors', obj=self)
        if threshold.shape[0] != self.n_items:
            raise ValidationError(
                "Number of threshold values (%d) does not match number of "
                "input vectors (%d)." % (threshold.shape[0], self.n_items),
                attr='threshold', obj=self)

        # --- Set parameters
        self.out_conns = []  # Used in `add_threshold_to_output`
        # Used in `add_threshold_to_output`
        self.default_vector_inhibit_conns = []
        self.thresh_ens = None  # Will hold thresholded outputs
        self.is_wta = False
        self._inhib_scale = 1.5

        # -- Create the core network
        with self, self.am_ens_config:
            self.bias_node = nengo.Node(output=1)
            self.elem_input = nengo.Node(
                size_in=self.n_items, label="element input")
            self.elem_output = nengo.Node(
                size_in=self.n_items, label="element output")
            self.utilities = self.elem_output

            self.am_ensembles = []
            label_prefix = "" if label is None else label + "_"
            filt_scale = 15
            filt_step_func = lambda x: filtered_step(x, 0.0, scale=filt_scale)
            for i in range(self.n_items):
                e = nengo.Ensemble(n_neurons, 1, label=label_prefix + str(i))
                self.am_ensembles.append(e)

                # Connect input and output nodes
                nengo.Connection(self.bias_node, e, transform=-threshold[i])
                nengo.Connection(self.elem_input[i], e)
                nengo.Connection(
                    e, self.elem_output[i], function=filt_step_func)

            if inhibitable:
                # Input node for inhibitory gating signal (if enabled)
                self.inhibit = nengo.Node(size_in=1, label="inhibit")
                nengo.Connection(self.inhibit, self.elem_input,
                                 transform=-np.ones((self.n_items, 1))
                                 * self._inhib_scale)
                # Note: We can use a decoded connection here because all the
                # am_ensembles have [1] encoders
            else:
                self.inhibit = None

            self.thresh_bias = None
            self.thresholded_utilities = None

        self.add_input_mapping("input", input_vectors, input_scales)
        self.add_output_mapping("output", output_vectors)

    @property
    def am_ens_config(self):
        """(Config) Defaults for associative memory ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble, nengo.Connection)
        cfg[nengo.Ensemble].update({
            'radius': 1,
            'intercepts': Exponential(self.exp_scale, 0.0, 1.0),
            'encoders': Choice([[1]]),
            'eval_points': Uniform(0.0, 1.0),
            'n_eval_points': self.n_eval_points,
        })
        cfg[nengo.Connection].synapse = None
        return cfg

    @property
    def default_ens_config(self):
        """(Config) Defaults for other ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble)
        cfg[nengo.Ensemble].update({
            'radius': 1,
            'intercepts': Exponential(self.exp_scale, 0.0, 1.0),
            'encoders': Choice([[1]]),
            'eval_points': Uniform(0.0, 1.0),
            'n_eval_points': self.n_eval_points,
        })
        return cfg

    @property
    def thresh_ens_config(self):
        """(Config) Defaults for threshold ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble)
        cfg[nengo.Ensemble].update({
            'radius': 1,
            'intercepts': Uniform(0.5, 1.0),
            'encoders': Choice([[1]]),
            'eval_points': Uniform(0.75, 1.1),
            'n_eval_points': self.n_eval_points,
        })
        return cfg

    @with_self
    def add_input_mapping(self, name, input_vectors, input_scales=1.0):
        """Adds a set of input vectors to the associative memory network.

        Creates a transform with the given input vectors between the
        a named input node and associative memory element input to enable the
        inputs to be mapped onto ensembles of the Associative Memory.

        Parameters
        ----------
        name: str
            Name to use for the input node. This name will be used as the name
            of the attribute for the associative memory network.
        input_vectors: array_like
            The list of vectors to be compared against.
        input_scales: float or array_like, optional (Default: 1.0)
            Scaling factor to apply on each of the input vectors. Note that it
            is possible to scale each vector independently.
        """
        # --- Put arguments in canonical form
        n_vectors, d_vectors = input_vectors.shape
        if is_iterable(input_vectors):
            input_vectors = np.array(input_vectors, ndmin=2)
        if not is_iterable(input_scales):
            input_scales = input_scales * np.ones((1, n_vectors))
        else:
            input_scales = np.array(input_scales, ndmin=2)

        # --- Check some preconditions
        if input_scales.shape[1] != n_vectors:
            raise ValidationError("Number of input_scale values (%d) does not "
                                  "match number of input vectors (%d)."
                                  % (input_scales.shape[1], n_vectors),
                                  attr='input_scales')
        if hasattr(self, name):
            raise ValidationError("Name '%s' already exists as a node in the "
                                  "associative memory." % name, attr='name')

        # --- Finally, make the input node and connect it
        in_node = nengo.Node(size_in=d_vectors, label=name)
        setattr(self, name, in_node)
        nengo.Connection(in_node, self.elem_input,
                         synapse=None,
                         transform=input_vectors * input_scales.T)

    @with_self
    def add_output_mapping(self, name, output_vectors):
        """Adds another output to the associative memory network.

        Creates a transform with the given output vectors between the
        associative memory element output and a named output node to enable the
        selection of output vectors by the associative memory.

        Parameters
        ----------
        name: str
            Name to use for the output node. This name will be used as
            the name of the attribute for the associative memory network.
        output_vectors: array_like
            The list of vectors to be produced for each match.
        """
        # --- Put arguments in canonical form
        if is_iterable(output_vectors):
            output_vectors = np.array(output_vectors, ndmin=2)

        # --- Check preconditions
        if hasattr(self, name):
            raise ValidationError("Name '%s' already exists as a node in the "
                                  "associative memory." % name, attr='name')

        # --- Make the output node and connect it
        output = nengo.Node(size_in=output_vectors.shape[1], label=name)
        setattr(self, name, output)

        if self.thresh_ens is not None:
            c = nengo.Connection(self.thresh_ens.output, output,
                                 synapse=None, transform=output_vectors.T)
        else:
            c = nengo.Connection(self.elem_output, output,
                                 synapse=None, transform=output_vectors.T)
        self.out_conns.append(c)

    @with_self
    def add_default_output_vector(self, output_vector, output_name='output',
                                  n_neurons=50, min_activation_value=0.5):
        """Adds a default output vector to the associative memory network.

        The default output vector is chosen if the input matches none of
        the given input vectors.

        Parameters
        ----------
        output_vector: array_like
            The vector to be produced if the input value matches none of
            the vectors in the input vector list.
        output_name: str, optional (Default: 'output')
            The name of the input to which the default output vector
            should be applied.
        n_neurons: int, optional (Default: 50)
            Number of neurons to use for the default output vector ensemble.
        min_activation_value: float, optional (Default: 0.5)
            Minimum activation value (i.e. threshold) to use to disable
            the default output vector.
        """
        with self.default_ens_config:
            default_vector_ens = nengo.Ensemble(
                n_neurons, 1, label="Default %s vector" % output_name)
            nengo.Connection(self.bias_node, default_vector_ens, synapse=None)

            tr = -(1.0 / min_activation_value) * np.ones((1, self.n_items))
            if self.thresh_ens is not None:
                c = nengo.Connection(
                    self.thresh_ens.output, default_vector_ens, transform=tr)
            else:
                c = nengo.Connection(
                    self.elem_output, default_vector_ens, transform=tr)

            # Add the output connection to the output connection list
            self.default_vector_inhibit_conns.append(c)

            # Make new output class attribute and connect to it
            output = getattr(self, output_name)
            nengo.Connection(default_vector_ens, output,
                             transform=np.array(output_vector, ndmin=2).T,
                             synapse=None)

            if self.inhibit is not None:
                nengo.Connection(self.inhibit, default_vector_ens,
                                 transform=-self._inhib_scale, synapse=None)

    @with_self
    def add_wta_network(self, inhibit_scale=1.5, inhibit_synapse=0.005):
        """Add a winner-take-all (WTA) network to associative memory output.

        Parameters
        ----------
        inhibit_scale: float, optional (Default: 1.5)
            Mutual inhibition scaling factor.
        inhibit_synapse: float, optional (Default: 0.005)
            Mutual inhibition synapse time constant.
        """
        if not self.is_wta:
            nengo.Connection(self.elem_output, self.elem_input,
                             synapse=inhibit_synapse,
                             transform=((np.eye(self.n_items) - 1) *
                                        inhibit_scale))
            self.is_wta = True
        else:
            warnings.warn("AssociativeMemory network is already configured "
                          "with a WTA network. Additional `add_wta_network` "
                          "calls are ignored.")

    @with_self
    def add_threshold_to_outputs(self, n_neurons=50, inhibit_scale=10):
        """Adds a thresholded output to the associative memory.

        Parameters
        ----------
        n_neurons: int, optional (Default: 50)
            Number of neurons to use for the default output vector ensemble.
        inhibit_scale: float, optional (Default: 10)
            Mutual inhibition scaling factor.
        """
        if self.thresh_ens is not None:
            warnings.warn("AssociativeMemory network is already configured "
                          "with thresholded outputs. Additional "
                          "`add_threshold_to_output` calls are ignored.")
            return

        with self.thresh_ens_config:
            self.thresh_bias = EnsembleArray(
                n_neurons, self.n_items, label='thresh_bias')
            self.thresh_ens = EnsembleArray(
                n_neurons, self.n_items, label='thresh_ens')
            self.thresholded_utilities = self.thresh_ens.output

            nengo.Connection(self.bias_node, self.thresh_bias.input,
                             transform=np.ones((self.n_items, 1)),
                             synapse=None)
            nengo.Connection(self.bias_node, self.thresh_ens.input,
                             transform=np.ones((self.n_items, 1)),
                             synapse=None)
            nengo.Connection(self.elem_output, self.thresh_bias.input,
                             transform=-inhibit_scale)
            nengo.Connection(self.thresh_bias.output, self.thresh_ens.input,
                             transform=-inhibit_scale)

            # Reroute connections from elem_output to be connections
            # from thresh_ens.output
            def reroute(conn):
                c = nengo.Connection(self.thresh_ens.output, conn.post,
                                     transform=conn.transform,
                                     synapse=conn.synapse)
                self.connections.remove(conn)
                return c
            self.default_vector_inhibit_conns = [
                reroute(c) for c in self.default_vector_inhibit_conns]
            self.out_conns = [reroute(c) for c in self.out_conns]

            # Make inhibitory connection if inhibit option is set
            if self.inhibit is not None:
                for e in self.thresh_ens.ensembles:
                    nengo.Connection(
                        self.inhibit, e, transform=-self._inhib_scale,
                        synapse=None)
