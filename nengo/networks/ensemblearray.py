import warnings

import numpy as np

from nengo.connection import Connection
from nengo.dists import Samples
from nengo.ensemble import Ensemble
from nengo.exceptions import ValidationError
from nengo.neurons import Direct
from nengo.network import Network
from nengo.node import Node
from nengo.utils.network import with_self
from nengo.utils.numpy import is_iterable


class EnsembleArray(Network):
    """An array of ensembles.

    This acts, in some ways, like a single high-dimensional ensemble,
    but actually consists of many sub-ensembles, each one representing
    a separate dimension. This tends to be much faster to create
    and can be more accurate than having one huge high-dimensional ensemble.
    However, since the neurons represent different dimensions separately,
    we cannot compute nonlinear interactions between those dimensions.

    Note that in addition to the parameters below, parameters affecting
    all of the sub-ensembles can be passed to the ensemble array.
    For example:

    .. testcode::

       ea = nengo.networks.EnsembleArray(20, 2, radius=1.5)

    creates an ensemble array with 2 sub-ensembles, each with 20 neurons,
    and a radius of 1.5.

    Parameters
    ----------
    n_neurons : int
        The number of neurons in each sub-ensemble.
    n_ensembles : int
        The number of sub-ensembles to create.
    ens_dimensions : int, optional
        The dimensionality of each sub-ensemble.
    label : str, optional
        A name to assign this EnsembleArray.
        Used for visualization and debugging.
    seed : int, optional
        Random number seed that will be used in the build step.
    add_to_container : bool, optional
        Determines if this network will be added to the current container.
        If None, this network will be added to the network at the top of the
        ``Network.context`` stack unless the stack is empty.

    Attributes
    ----------
    dimensions_per_ensemble : int
        The dimensionality of each sub-ensemble.
    ea_ensembles : list
        The sub-ensembles in the ensemble array.
    input : Node
        A node that provides input to all of the ensembles in the array.
    n_ensembles : int
        The number of sub-ensembles to create.
    n_neurons_per_ensemble : int
        The number of neurons in each sub-ensemble.
    neuron_input : Node or None
        A node that provides input to all the neurons in the ensemble array.
        None unless created in `~.EnsembleArray.add_neuron_input`.
    neuron_output : Node or None
        A node that gathers neural output from all the neurons in the ensemble
        array. None unless created in `~.EnsembleArray.add_neuron_output`.
    output : Node
        A node that gathers decoded output from all of the ensembles
        in the array.
    """

    def __init__(
        self,
        n_neurons,
        n_ensembles,
        ens_dimensions=1,
        label=None,
        seed=None,
        add_to_container=None,
        **ens_kwargs
    ):
        if "dimensions" in ens_kwargs:
            raise ValidationError(
                "'dimensions' is not a valid argument to EnsembleArray. "
                "To set the number of ensembles, use 'n_ensembles'. To set "
                "the number of dimensions per ensemble, use 'ens_dimensions'.",
                attr="dimensions",
                obj=self,
            )

        super().__init__(label, seed, add_to_container)

        for param in ens_kwargs:
            if is_iterable(ens_kwargs[param]):
                ens_kwargs[param] = Samples(ens_kwargs[param])

        self.config[Ensemble].update(ens_kwargs)

        label_prefix = "" if label is None else label + "_"

        self.n_neurons_per_ensemble = n_neurons
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_dimensions

        # These may be set in add_neuron_input and add_neuron_output
        self.neuron_input, self.neuron_output = None, None

        self.ea_ensembles = []

        with self:
            self.input = Node(size_in=self.dimensions, label="input")

            for i in range(n_ensembles):
                e = Ensemble(
                    n_neurons,
                    self.dimensions_per_ensemble,
                    label="%s%d" % (label_prefix, i),
                )
                Connection(
                    self.input[i * ens_dimensions : (i + 1) * ens_dimensions],
                    e,
                    synapse=None,
                )
                self.ea_ensembles.append(e)

        self.add_output("output", function=None)

    @property
    def dimensions(self):
        """(int) Dimensionality of the ensemble array."""
        return self.n_ensembles * self.dimensions_per_ensemble

    @with_self
    def add_neuron_input(self):
        """Adds a node that provides input to the neurons of all ensembles.

        Direct neuron input is useful for inhibiting the activity of all
        neurons in the ensemble array.

        This node is accessible through the 'neuron_input' attribute
        of this ensemble array.
        """
        if self.neuron_input is not None:
            warnings.warn("neuron_input already exists. Returning.")
            return self.neuron_input

        if isinstance(self.ea_ensembles[0].neuron_type, Direct):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot give neuron input to Direct neurons.",
                attr="ea_ensembles[0].neuron_type",
                obj=self,
            )

        self.neuron_input = Node(
            size_in=self.n_neurons_per_ensemble * self.n_ensembles, label="neuron_input"
        )

        for i, ens in enumerate(self.ea_ensembles):
            Connection(
                self.neuron_input[
                    i
                    * self.n_neurons_per_ensemble : (i + 1)
                    * self.n_neurons_per_ensemble
                ],
                ens.neurons,
                synapse=None,
            )
        return self.neuron_input

    @with_self
    def add_neuron_output(self):
        """Adds a node that collects the neural output of all ensembles.

        Direct neuron output is useful for plotting the spike raster of
        all neurons in the ensemble array.

        This node is accessible through the 'neuron_output' attribute
        of this ensemble array.
        """
        if self.neuron_output is not None:
            warnings.warn("neuron_output already exists. Returning.")
            return self.neuron_output

        if isinstance(self.ea_ensembles[0].neuron_type, Direct):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot get neuron output from Direct neurons.",
                attr="ea_ensembles[0].neuron_type",
                obj=self,
            )

        self.neuron_output = Node(
            size_in=self.n_neurons_per_ensemble * self.n_ensembles,
            label="neuron_output",
        )

        for i, ens in enumerate(self.ea_ensembles):
            Connection(
                ens.neurons,
                self.neuron_output[
                    i
                    * self.n_neurons_per_ensemble : (i + 1)
                    * self.n_neurons_per_ensemble
                ],
                synapse=None,
            )
        return self.neuron_output

    @with_self
    def add_output(self, name, function, synapse=None, **conn_kwargs):
        """Adds a node that collects the decoded output of all ensembles.

        By default, this is called once in ``__init__`` with ``function=None``.
        However, this can be called multiple times with different functions,
        similar to the way in which an ensemble can be connected to many
        downstream ensembles with different functions.

        Note that in addition to the parameters below, parameters affecting
        all of the connections from the sub-ensembles to the new node
        can be passed to this function. For example:

        .. testcode::

           ea.add_output('output', None, solver=nengo.solvers.Lstsq())

        creates a new output with the decoders of each connection solved for
        with the `.Lstsq` solver.

        Parameters
        ----------
        name : str
            The name of the output. This will also be the name of the attribute
            set on the ensemble array.
        function : callable or iterable of callables
            The function to compute across the connection from sub-ensembles
            to the new output node. If function is an iterable, it must be
            an iterable consisting of one function for each sub-ensemble.
        synapse : Synapse, optional
            The synapse model with which to filter the connections from
            sub-ensembles to the new output node. This is kept separate from
            the other ``conn_kwargs`` because this defaults to None rather
            than the default synapse model. In almost all cases the synapse
            should stay as None, and synaptic filtering should be performed in
            the connection from the output node.
        """
        dims_per_ens = self.dimensions_per_ensemble

        # get output size for each ensemble
        sizes = np.zeros(self.n_ensembles, dtype=int)

        if is_iterable(function) and all(callable(f) for f in function):
            if len(list(function)) != self.n_ensembles:
                raise ValidationError(
                    "Must have one function per ensemble", attr="function"
                )

            for i, func in enumerate(function):
                sizes[i] = np.asarray(func(np.zeros(dims_per_ens))).size
        elif callable(function):
            sizes[:] = np.asarray(function(np.zeros(dims_per_ens))).size
            function = [function] * self.n_ensembles
        elif function is None:
            sizes[:] = dims_per_ens
            function = [None] * self.n_ensembles
        else:
            raise ValidationError(
                "'function' must be a callable, list of callables, or None",
                attr="function",
            )

        output = Node(output=None, size_in=sizes.sum(), label=name)
        setattr(self, name, output)

        indices = np.zeros(len(sizes) + 1, dtype=int)
        indices[1:] = np.cumsum(sizes)
        for i, e in enumerate(self.ea_ensembles):
            Connection(
                e,
                output[indices[i] : indices[i + 1]],
                function=function[i],
                synapse=synapse,
                **conn_kwargs,
            )

        return output
