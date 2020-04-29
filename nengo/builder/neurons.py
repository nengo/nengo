import inspect

import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.dists import Distribution
from nengo.exceptions import BuildError
from nengo.neurons import (
    NeuronType,
    RegularSpiking,
    _Spiking,
)
from nengo.utils.numpy import is_array_like


class SimNeurons(Operator):
    """Set a neuron model output for the given input current.

    Implements ``neurons.step_math(dt, J, output, *states)``.

    Parameters
    ----------
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step_math`` function.
    J : Signal
        The input current.
    output : Signal
        The neuron output signal that will be set.
    states : list, optional
        A list of additional neuron state signals set by ``step_math``.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    J : Signal
        The input current.
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step_math`` function.
    output : Signal
        The neuron output signal that will be set.
    states : list
        A list of additional neuron state signals set by ``step_math``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[output] + states``
    2. incs ``[]``
    3. reads ``[J]``
    4. updates ``[]``
    """

    def __init__(self, neurons, J, output, states=None, tag=None):
        super().__init__(tag=tag)
        self.neurons = neurons

        self.sets = [output] + ([] if states is None else states)
        self.incs = []
        self.reads = [J]
        self.updates = []

    @property
    def J(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    @property
    def states(self):
        return self.sets[1:]

    def _descstr(self):
        return "%s, %s, %s" % (self.neurons, self.J, self.output)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        argspec = inspect.getfullargspec(self.neurons.step_math)
        if "rng" in argspec.args:

            def step_simneurons_withrng():
                self.neurons.step_math(dt, J, output, rng, *states)

            return step_simneurons_withrng
        else:

            def step_simneurons():
                self.neurons.step_math(dt, J, output, *states)

            return step_simneurons


@Builder.register(NeuronType)
def build_neurons(model, neurontype, neurons):
    """Builds a `.NeuronType` object into a model.

    This build function works with any `.NeuronType` that does not require
    extra state, like `.RectifiedLinear` and `.LIFRate`. This function adds a
    `.SimNeurons` operator connecting the input current to the
    neural output signals.

    Parameters
    ----------
    model : Model
        The model to build into.
    neurontype : NeuronType
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.NeuronType` instance.
    """
    dtype = model.sig[neurons]["in"].dtype
    n_neurons = neurons.size_in
    state_init = neurontype.make_neuron_state(n_neurons, model.dt, dtype=dtype)

    states = []
    for key, init in state_init.items():
        if key in model.sig[neurons]:
            raise BuildError("State name %r overlaps with existing signal name" % key)

        if isinstance(init, Distribution):
            raise NotImplementedError()
        elif is_array_like(init):
            init = np.asarray(init, dtype=dtype)
            if init.ndim == 0:
                init = init * np.ones(n_neurons, dtype=dtype)
            elif init.ndim == 1 and init.size != n_neurons or init.ndim > 1:
                raise BuildError(
                    "State init array must be 0-D, or 1-D of length `n_neurons`"
                )
        else:
            raise BuildError("State init must be a distribution or array-like")

        model.sig[neurons][key] = Signal(
            initial_value=init, shape=(n_neurons,), name="%s.%s" % (neurons, key)
        )
        states.append(model.sig[neurons][key])

    model.add_op(
        SimNeurons(
            neurons=neurontype,
            J=model.sig[neurons]["in"],
            output=model.sig[neurons]["out"],
            states=states,
        )
    )


@Builder.register(_Spiking)
def build_spiking(model, neuron_type, neurons):
    """Builds a `._Spiking` object into a model.

    This builder delegates most of the process to the base neuron type builder,
    then modifies the `.SimNeurons` operator made in the base builder.

    Parameters
    ----------
    model : Model
        The model to build into.
    neuron_type : subclass of `._Spiking`
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.LIF` instance.
    """
    model.build(neuron_type.base_type, neurons)
    op = model.operators[-1]
    assert isinstance(op, SimNeurons) and op.neurons == neuron_type.base_type
    op.neurons = neuron_type
    return op


@Builder.register(RegularSpiking)
def build_regular_spiking(model, reg, neurons):
    """Builds a `.RegularSpiking` object into a model.

    This builder delegates most of the process to the base neuron type builder,
    then sets up an additional signal. We then modify the `.SimNeurons`
    operator made in the base builder.

    Parameters
    ----------
    model : Model
        The model to build into.
    reg : RegularSpiking
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.LIF` instance.
    """
    op = build_spiking(model, reg, neurons)

    # set voltage to 0.5 to be between positive and negative spike thresholds of 1 and 0
    model.sig[neurons]["voltage"] = Signal(
        0.5 * np.ones(neurons.size_in), name="%s.voltage" % neurons
    )
    # insert right after `output` so that this is the first state
    op.sets.insert(1, model.sig[neurons]["voltage"])
