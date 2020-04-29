from collections import OrderedDict

import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.dists import Distribution
from nengo.exceptions import BuildError
from nengo.neurons import NeuronType
from nengo.rc import rc
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

        self.states = OrderedDict() if states is None else states

        self.sets = [output] + list(self.states.values())
        self.incs = []
        self.reads = [J]
        self.updates = []

    @property
    def J(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    def _descstr(self):
        return "%s, %s, %s" % (self.neurons, self.J, self.output)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        states = {name: signals[sig] for name, sig in self.states.items()}

        def step_simneurons():
            self.neurons.step_math(dt, J, output, **states)

        return step_simneurons


def get_neuron_states(model, neurontype, neurons, dtype=None):
    """Get the initial neuron state values for the neuron type."""
    dtype = rc.float_dtype if dtype is None else dtype
    ens = neurons.ensemble
    rng = np.random.RandomState(model.seeds[ens] + 1)
    n_neurons = neurons.size_in

    if isinstance(ens.initial_phase, Distribution):
        phases = ens.initial_phase.sample(n_neurons, rng=rng)
    else:
        phases = ens.initial_phase

    if phases.ndim != 1 or phases.size != n_neurons:
        raise BuildError("`initial_phase` array must be 1-D of length `n_neurons`")

    return neurontype.make_neuron_state(phases, model.dt, dtype=dtype)


def sample_state_init(state_init, n_neurons, dtype=None):
    """Sample a state init value, ensuring the correct size. """
    dtype = rc.float_dtype if dtype is None else dtype

    if isinstance(state_init, Distribution):
        raise NotImplementedError()
    elif is_array_like(state_init):
        state_init = np.asarray(state_init, dtype=dtype)
        if state_init.ndim == 0:
            state_init = state_init * np.ones(n_neurons, dtype=dtype)
        elif (
            state_init.ndim == 1 and state_init.size != n_neurons or state_init.ndim > 1
        ):
            raise BuildError(
                "State init array must be 0-D, or 1-D of length `n_neurons`"
            )
    else:
        raise BuildError("State init must be a distribution or array-like")

    return state_init


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
    n_neurons = neurons.size_in
    state_init = get_neuron_states(model, neurontype, neurons)

    states = OrderedDict()
    for key, init in state_init.items():
        if key in model.sig[neurons]:
            raise BuildError("State name %r overlaps with existing signal name" % key)

        states[key] = model.sig[neurons][key] = Signal(
            initial_value=sample_state_init(init, n_neurons),
            shape=(n_neurons,),
            name="%s.%s" % (neurons, key),
        )

    model.add_op(
        SimNeurons(
            neurons=neurontype,
            J=model.sig[neurons]["in"],
            output=model.sig[neurons]["out"],
            states=states,
        )
    )
