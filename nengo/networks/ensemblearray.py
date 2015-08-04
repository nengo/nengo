import warnings

import numpy as np

import nengo
from nengo.utils.compat import is_iterable, range
from nengo.utils.network import with_self


class EnsembleArray(nengo.Network):
    """An array of ensembles.

    This acts, in some ways, like a single high-dimensional ensemble,
    but actually consists of many sub-ensembles, each one representing
    a separate dimension. This tends to be much faster to create
    and can be more accurate than having one huge high-dimensional ensemble.
    However, since the neurons represent different dimensions separately,
    we cannot compute nonlinear interactions between those dimensions.

    Parameters
    ----------
    n_neurons : int
        The number of neurons in each sub-ensemble.
    n_ensembles : int
        The number of sub-ensembles to create.
    ens_dimensions: int, optional
        The dimensionality of each sub-ensemble. Default: 1.
    neuron_nodes : bool, optional
        Whether to create a node that provides each access to each individual
        neuron, typically for the purpose of inibiting the entire
        EnsembleArray. Default: False.
    label : str, optional
        A name to assign this EnsmbleArray.
        Used for visualization and debugging.
    seed : int, optional
        Random number seed that will be used in the build step.
    add_to_container : bool, optional
        Whether this network will be added to the current context.

    Additional parameters for each sub-ensemble can be passed through
    ``**ens_kwargs``.
    """

    def __init__(self, n_neurons, n_ensembles, ens_dimensions=1,
                 neuron_nodes=False, label=None, seed=None,
                 add_to_container=None, **ens_kwargs):
        if "dimensions" in ens_kwargs:
            raise TypeError(
                "'dimensions' is not a valid argument to EnsembleArray. "
                "To set the number of ensembles, use 'n_ensembles'. To set "
                "the number of dimensions per ensemble, use 'ens_dimensions'.")

        super(EnsembleArray, self).__init__(label, seed, add_to_container)

        self.config[nengo.Ensemble].update(ens_kwargs)

        label_prefix = "" if label is None else label + "_"

        self.n_neurons = n_neurons
        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_dimensions

        self.ea_ensembles = []

        with self:
            self.input = nengo.Node(size_in=self.dimensions, label="input")

            if neuron_nodes:
                self.neuron_input = nengo.Node(
                    size_in=n_neurons * n_ensembles, label="neuron_input")
                self.neuron_output = nengo.Node(
                    size_in=n_neurons * n_ensembles, label="neuron_output")

            for i in range(n_ensembles):
                e = nengo.Ensemble(
                    n_neurons, self.dimensions_per_ensemble,
                    label=label_prefix + str(i))

                nengo.Connection(self.input[i * ens_dimensions:
                                            (i + 1) * ens_dimensions],
                                 e, synapse=None)

                if (neuron_nodes
                        and not isinstance(e.neuron_type, nengo.Direct)):
                    nengo.Connection(self.neuron_input[i * n_neurons:
                                                       (i + 1) * n_neurons],
                                     e.neurons, synapse=None)
                    nengo.Connection(e.neurons,
                                     self.neuron_output[i * n_neurons:
                                                        (i + 1) * n_neurons],
                                     synapse=None)

                self.ea_ensembles.append(e)

            if neuron_nodes and isinstance(e.neuron_type, nengo.Direct):
                warnings.warn("Creating neuron nodes in an EnsembleArray"
                              " with Direct neurons")

        self.add_output('output', function=None)

    @with_self
    def add_output(self, name, function, synapse=None, **conn_kwargs):
        dims_per_ens = self.dimensions_per_ensemble

        # get output size for each ensemble
        sizes = np.zeros(self.n_ensembles, dtype=int)

        if is_iterable(function) and all(callable(f) for f in function):
            if len(function) != self.n_ensembles:
                raise ValueError("Must have one function per ensemble")

            for i, func in enumerate(function):
                sizes[i] = np.asarray(func(np.zeros(dims_per_ens))).size
        elif callable(function):
            sizes[:] = np.asarray(function(np.zeros(dims_per_ens))).size
            function = [function] * self.n_ensembles
        elif function is None:
            sizes[:] = dims_per_ens
            function = [None] * self.n_ensembles
        else:
            raise ValueError(
                "'function' must be a callable, list of callables, or 'None'")

        output = nengo.Node(output=None, size_in=sizes.sum(), label=name)
        setattr(self, name, output)

        indices = np.zeros(len(sizes) + 1, dtype=int)
        indices[1:] = np.cumsum(sizes)
        for i, e in enumerate(self.ea_ensembles):
            nengo.Connection(
                e, output[indices[i]:indices[i+1]], function=function[i],
                synapse=synapse, **conn_kwargs)

        return output

    @property
    def dimensions(self):
        return self.n_ensembles * self.dimensions_per_ensemble
