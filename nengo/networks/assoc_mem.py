import warnings

import numpy as np

import nengo
from .ensemblearray import EnsembleArray
from nengo.dists import Choice, Exponential, Uniform
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_iterable, range
from nengo.utils.network import with_self


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

    # Prefix and suffix tags to be used when creating the corresponding output
    # nodes. E.g. Adding a 'utitilies' node to the 'output' output will use
    # the 'utility_output_suffix' in the name, creating a node called
    # 'output_utilities'
    cleanup_output_prefix = 'cleaned'
    default_ens_suffix = 'default_ens'
    utility_output_suffix = 'utilities'

    def __init__(self, input_vectors, output_vectors=None,  # noqa: C901
                 n_neurons=50, threshold=0.3, input_scales=1.0,
                 inhibitable=False, label=None, **network_kwargs):
        super(AssociativeMemory, self).__init__(label=label, **network_kwargs)

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
        self.thresholds = threshold

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
        self.is_wta = False
        self._inhib_scale = 1.5
        self._output_vectors = {}
        self._default_output_vectors = {}

        # -- Create the core network
        with self, self.am_ens_config:
            self.bias_node = nengo.Node(output=1)
            self.ens_input = nengo.Node(
                size_in=self.n_items, label="ensembles input")
            self.ens_utilities = nengo.Node(
                size_in=self.n_items, label="ensembles utilities")

            self.am_ensembles = []
            label_prefix = "" if label is None else label + "_"

            for i in range(self.n_items):
                e = nengo.Ensemble(n_neurons, 1, label=label_prefix + str(i))
                self.am_ensembles.append(e)

                # Connect input and output nodes
                nengo.Connection(self.bias_node, e, transform=-threshold[i])
                nengo.Connection(self.ens_input[i], e, synapse=None)
                nengo.Connection(e, self.ens_utilities[i], synapse=None)

            if inhibitable:
                # Input node for inhibitory gating signal (if enabled)
                self.inhibit = nengo.Node(size_in=1, label="inhibit")
                nengo.Connection(self.inhibit, self.ens_input,
                                 transform=-np.ones((self.n_items, 1))
                                 * self._inhib_scale, synapse=None)
                # Note: We can use a decoded connection here because all the
                # am_ensembles have [1] encoders
            else:
                self.inhibit = None
        self.add_input_mapping("input", input_vectors, input_scales)
        self.add_output_mapping("output", output_vectors)

    @staticmethod
    def linear_func(x_shift=0.0, x_scale=1.0):
        """Returns a linear mapping function.

        Returns a linear mapping function for ease of use when creating output
        mappings. The returned function takes in a non-optional 'x' parameter,
        as well as optional parameters listed below.

        Parameters
        ----------
        x_shift: float, optional
            Amount to shift the linear map on the 'x'-axis.
            E.g. If the value of x_shift = 0.3, then the output of the
                 mapping function would be 0 at x == 0.3.
        x_scale: float, optional
            Scaling factor to be applied to the 'x' input. Affects the slope
            of the linear mapping function.
        """
        return lambda x, x_shift=x_shift, x_scale=x_scale: \
            (x_scale * (x - x_shift))

    def threshold_shifted_linear_funcs(self, x_shift=0.0, x_scale=1.0):
        """Returns a list of threshold-shifted linear mapping function.

        Produces a list of linear mapping functions shifted by the current
        threshold values (one threshold for each of the assoc mem items).
        Useful for generating linear output mappings that are shifted
        appropriately with the associative memory threshold values.
        (see function usage in add_output_mapping)

        Parameters
        ----------
        x_shift: float, optional
            Amount to shift the linear map on the 'x'-axis, pre-shifted by the
            threshold values for each ensemble.
            E.g. If the value of x_shift = 0.3, then the output of the linear
                 mapping function would be 0 at x == (0.3 - threshold).
        x_scale: float, optional
            Scaling factor to be applied to the 'x' input. Affects the slope
            of the linear mapping function.
        """
        return [self.linear_func(-threshold + x_shift, x_scale)
                for threshold in self.thresholds]

    @staticmethod
    def filtered_step_func(x_shift=0.0, x_scale=15.0, step_val=1):
        """Returns a filtered step mapping function.

        Returns a filtered step mapping function for ease of use when creating
        output mappings. The returned function takes in a non-optional 'x'
        parameter, as well as optional parameters listed below.

        Parameters
        ----------
        x_shift: float, optional
            Amount to shift the linear map on the 'x'-axis.
            E.g. If the value of x_shift = 0.3, then the output of the
                 mapping function would be 0 at x == 0.3.
        x_scale: float, optional
            Scaling factor to be applied to the 'x' input. Affects the
            sharpness of the filtered step edge. Larger values produce a
            sharper edge. Set to a negative value to flip the mapping
            function about the 'y' axis at x == x_shift.
        step_val: float, optional
            Maximal value of the filtered step function.
        """
        return (lambda x, x_shift=x_shift, x_scale=x_scale, step_val=step_val:
                np.maximum(-1 / np.exp(x_scale * (x - x_shift)) + 1, 0) *
                step_val)

    @staticmethod
    def step_func(x_shift=0.0, x_scale=1.0, step_val=1):
        """Returns a step mapping function.

        Returns a step mapping function for ease of use when creating
        output mappings. The returned function takes in a non-optional 'x'
        parameter, as well as optional parameters listed below.

        Parameters
        ----------
        x_shift: float, optional
            Amount to shift the linear map on the 'x'-axis.
            E.g. If the value of x_shift = 0.3, then the output of the
                 mapping function would be 0 at x == 0.3.
        x_scale: float, optional
            Scaling factor to be applied to the 'x' input. Set to a negative
            value to flip the mapping function about the 'y' axis at
            x == x_shift.
        step_val: float, optional
            Maximal value of the filtered step function.
        """
        return (lambda x, x_shift=x_shift, x_scale=x_scale, step_val=step_val:
                np.maximum(np.minimum(np.ceil(x_scale * (x - x_shift)), 1.0),
                           0.0))

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
            'intercepts': Uniform(0.25, 1.0),
            'encoders': Choice([[1]]),
            'eval_points': Uniform(0.75, 1.1),
            'n_eval_points': self.n_eval_points,
        })
        return cfg

    @property
    def cleanup_ens_config(self):
        """(Config) Defaults for threshold ensemble creation."""
        cfg = nengo.Config(nengo.Ensemble)
        cfg[nengo.Ensemble].update({
            'radius': 1,
            'intercepts': Uniform(0.25, 1.0),
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
        nengo.Connection(in_node, self.ens_input,
                         synapse=None,
                         transform=input_vectors * input_scales.T)

    @with_self
    def add_output_mapping(self, name, output_vectors, utility_map_func=None):
        """Adds another output to the associative memory network.

        Creates a transform with the given output vectors between the
        associative memory element output and a named output node to enable the
        selection of output vectors by the associative memory. A function can
        be provided to customize the mapping between the associative memory
        element utilities and the output vector utilities.

        Note: This function creates 2 nodes:
            - The specified output node with the name '<NAME>' given above.
            - A node named '<NAME>_utilities' to output the mapped output
              utilities.

        Parameters
        ----------
        name: str
            Name to use for the output node. This name will be used as
            the name of the attribute for the associative memory network.

        output_vectors: array_like
            The list of vectors to be produced for each match.

        utility_map_func: function or list of functions, optional
            The function used to map the utilities of the input vectors to the
            utility of the output vector. If not provided, will default to
            'self.filtered_step_func'.
            If a list of functions is provided, each function will be used
            for its respective neural ensemble.
            Note: The mapping function is threshold independent, and is aligned
                  such that the input ('x') to the mapping function is 0 when
                  the input utility is at the threshold value.
                  I.e. output_utility = map_func(input_utility - threshold)
        """
        # --- Put arguments in canonical form
        if is_iterable(output_vectors):
            output_vectors = np.array(output_vectors, ndmin=2)
        self._output_vectors[name] = output_vectors

        # --- Check preconditions
        if hasattr(self, name):
            raise ValidationError("Name '%s' already exists as a node in the "
                                  "associative memory." % name, attr='name')
        if utility_map_func is None:
            utility_map_func = self.filtered_step_func()
        if not is_iterable(utility_map_func):
            utility_map_funcs = [utility_map_func] * self.n_items
        else:
            utility_map_funcs = utility_map_func

        # --- Make the output node, and output utilities node and connect them
        #     Note: The output utilities make it easy to probe the raw output
        #           utilities and make it handy to do the appropriate
        #           connections in add_default_output_vector
        output = nengo.Node(size_in=output_vectors.shape[1], label=name)
        setattr(self, name, output)

        utility_node_name = '_'.join([name, self.utility_output_suffix])
        utility = nengo.Node(size_in=self.n_items, label=utility_node_name)
        setattr(self, utility_node_name, utility)

        for i, am_ens in enumerate(self.am_ensembles):
            nengo.Connection(am_ens, output, synapse=None,
                             transform=output_vectors[i, :, None],
                             function=utility_map_funcs[i])
            nengo.Connection(am_ens, utility[i], synapse=None,
                             function=utility_map_funcs[i])

    @with_self
    def add_default_output_vector(self, output_vector, output_name='output',
                                  n_neurons=50, min_activation_value=0.5,
                                  inhibit_scale=3.5):
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

        inhibit_scale: float, optional
            Scaling factor applied to the inhibitory connection used to disable
            the default output vector. It is recommended that this value be at
            least 1.0 / minimum(assoc memory activation thresholds), and that
            the minimum assoc memory activation threshold be at most 0.1.
        """
        default_ens_name = '_'.join([output_name, self.default_ens_suffix])
        self._default_output_vectors[output_name] = np.array(output_vector,
                                                             ndmin=2)

        # --- Check if default output vector has already been created for
        #     the desired output node
        if hasattr(self, default_ens_name):
            raise ValidationError("Default output vector already exists for "
                                  "output: '%s'." % output_name,
                                  attr='output_name')

        with self.default_ens_config:
            # --- Make the default vector ensemble, connect it to a bias node
            default_vector_ens = nengo.Ensemble(
                n_neurons, 1, label="Default %s vector" % output_name)
            setattr(self, default_ens_name, default_vector_ens)

            nengo.Connection(self.bias_node, default_vector_ens, synapse=None)

            # --- Get the output utilities and connect to the default vector
            #     ens
            tr = -(inhibit_scale) * np.ones((1, self.n_items))
            utility = getattr(self, '_'.join([output_name,
                                              self.utility_output_suffix]))
            nengo.Connection(utility, default_vector_ens, transform=tr)

            # Get the output class attribute and connect to it
            output_node = getattr(self, output_name)
            nengo.Connection(default_vector_ens, output_node,
                             transform=np.array(output_vector, ndmin=2).T,
                             synapse=None)

            # --- Create inhibitory connection if needed
            if self.inhibit is not None:
                nengo.Connection(self.inhibit, default_vector_ens,
                                 transform=-self._inhib_scale, synapse=None)

            # --- Connect default output vector to cleaned outputs
            #     (if available)
            cleanup_output_node = getattr(self,
                                          '_'.join([self.cleanup_output_prefix,
                                                    output_name]),
                                          None)
            if cleanup_output_node and (output_node != cleanup_output_node):
                nengo.Connection(default_vector_ens, cleanup_output_node,
                                 transform=np.array(output_vector, ndmin=2).T,
                                 synapse=None)

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
            nengo.Connection(self.ens_utilities, self.ens_input,
                             synapse=inhibit_synapse,
                             transform=((np.eye(self.n_items) - 1) *
                                        inhibit_scale))
            self.is_wta = True
        else:
            warnings.warn("AssociativeMemory network is already configured "
                          "with a WTA network. Additional `add_wta_network` "
                          "calls are ignored.")

    @with_self
    def add_cleanup_output(self, output_name='output', n_neurons=50,
                           inhibit_scale=3.5, replace_output=False):
        """Adds cleaned outputs to the associative memory network.

        Creates a doubled-inhibited ensemble structure to the desired assoc
        memory output to perform a cleanup operation on it. Note that using the
        'filtered_step_func' utility mapping performs a similar cleanup
        operation, but does not do a very good cleanup approximation for
        utility values near (+/- 0.2) the threshold value. This function
        adds the infrastructure needed to perform the cleanup operation across
        the entire range of output values, at the cost of two synaptic delays
        and adding (n_neurons * 2 * n_items) additional neurons to the network.

        Note: This function creates 2 nodes:
            - A node named 'cleaned_<OUTPUT_NAME>' that outputs the cleaned
              version of the output vectors.
            - A node named 'cleaned_<OUTPUT_NAME>_utilities' that outputs the
              utilities of the cleaned output vectors.

        Parameters
        ----------
        output_name: string, optional
            The name of the input to which the default output vector
            should be applied.

        n_neurons: int, optional
            Number of neurons to use for the ensembles used in the double-
            inhibited cleanup network.
        inhibit_scale: float, optional
            Scaling factor applied to the inhibitory connections between
            the ensembles. It is recommended that this value be at
            least 1.0 / minimum(assoc memory activation thresholds), and that
            the minimum assoc memory activation threshold be at most 0.1.

        replace_output: boolean, optional
            Flag to indicate whether or not to replace the output object
            (e.g. am.output) with the cleaned output node.
        """
        cleanup_output_name = '_'.join([self.cleanup_output_prefix,
                                        output_name])
        cleanup_utilities_name = '_'.join([self.cleanup_output_prefix,
                                           output_name,
                                           self.utility_output_suffix])
        output_utilities_name = '_'.join([output_name,
                                          self.utility_output_suffix])

        # --- Check if cleanup network has already been created for
        #     the desired output node
        if hasattr(self, cleanup_output_name):
            raise ValidationError("Cleanup output already exists for "
                                  "output: '%s'." % output_name,
                                  attr='output_name')

        with self.cleanup_ens_config:
            # --- Set up the double inhibited ensembles, and make the
            #     appropriate connections.
            self.bias_ens1 = EnsembleArray(
                n_neurons, self.n_items, label=output_name + '_bias_ens1')
            self.bias_ens2 = EnsembleArray(
                n_neurons, self.n_items, label=output_name + '_bias_ens2')

            utility = getattr(self, output_utilities_name)

            nengo.Connection(self.bias_node, self.bias_ens1.input,
                             transform=np.ones((self.n_items, 1)),
                             synapse=None)
            nengo.Connection(self.bias_node, self.bias_ens2.input,
                             transform=np.ones((self.n_items, 1)),
                             synapse=None)
            nengo.Connection(utility, self.bias_ens1.input,
                             transform=-inhibit_scale)
            nengo.Connection(self.bias_ens1.output, self.bias_ens2.input,
                             transform=-1.0)

            # --- Make the output node and connect it
            output_vectors = self._output_vectors[output_name]
            cleanup_output_node = nengo.Node(size_in=output_vectors.shape[1],
                                             label=cleanup_output_name)
            nengo.Connection(self.bias_ens2.output, cleanup_output_node,
                             transform=output_vectors.T, synapse=None)

            setattr(self, cleanup_output_name, cleanup_output_node)
            setattr(self, cleanup_utilities_name, self.bias_ens2.output)

            # --- Replace the original output node (pointer) if required
            if replace_output:
                setattr(self, output_name, cleanup_output_node)

            # --- Make inhibitory connection if inhibit option is set
            if self.inhibit is not None:
                for e in self.bias_ens2.ensembles:
                    nengo.Connection(
                        self.inhibit, e, transform=-self._inhib_scale,
                        synapse=None)

            # --- Connect default output vector to cleaned outputs
            #     (if available)
            default_vector_ens_name = '_'.join([output_name,
                                                self.default_ens_suffix])
            if hasattr(self, default_vector_ens_name):
                default_output_vectors = self._default_output_vectors[
                    output_name]
                nengo.Connection(getattr(self, default_vector_ens_name),
                                 cleanup_output_node,
                                 transform=default_output_vectors.T,
                                 synapse=None)
