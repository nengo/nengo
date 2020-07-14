import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.exceptions import BuildError
from nengo.neurons import NeuronType, RatesToSpikesNeuronType
from nengo.utils.numpy import is_array_like


class SimNeurons(Operator):
    """Set a neuron model output for the given input current.

    Implements ``neurons.step(dt, J, **state)``.

    Parameters
    ----------
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step`` function.
    J : Signal
        The input current.
    output : Signal
        The neuron output signal that will be set
    state : list, optional
        A list of additional neuron state signals set by ``step``.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    J : Signal
        The input current.
    neurons : NeuronType
        The `.NeuronType`, which defines a ``step`` function.
    output : Signal
        The neuron output signal that will be set.
    state : list
        A list of additional neuron state signals set by ``step``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[output] + state``
    2. incs ``[]``
    3. reads ``[J]``
    4. updates ``[]``
    """

    def __init__(self, neurons, J, output, state=None, tag=None):
        super().__init__(tag=tag)
        self.neurons = neurons

        self.sets = [output]
        self.incs = []
        self.reads = [J]
        self.updates = []

        self.state_idxs = {}
        self.state_extra = {}
        if state is not None:
            for name, obj in state.items():
                if isinstance(obj, Signal):
                    # The signals actually stored in `self.sets` can be modified by the
                    # optimizer. To allow this possibility, we store the index of the
                    # signal in the sets list instead of storing the signal itself.
                    self.state_idxs[name] = len(self.sets)
                    self.sets.append(obj)
                else:
                    # The only supported extra is a RandomState right now
                    assert isinstance(obj, np.random.RandomState)
                    self.state_extra[name] = obj

    @property
    def J(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    @property
    def state(self):
        return {key: self.sets[idx] for key, idx in self.state_idxs.items()}

    @property
    def _descstr(self):
        return "%s, %s" % (self.neurons, self.J)

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        state = {name: signals[sig] for name, sig in self.state.items()}
        state.update(self.state_extra)

        def step_simneurons():
            self.neurons.step(dt, J, output, **state)

        return step_simneurons


@Builder.register(NeuronType)
def build_neurons(model, neurontype, neurons, input_sig=None, output_sig=None):
    """Builds a `.NeuronType` object into a model.

    This function adds a `.SimNeurons` operator connecting the input current to the
    neural output signals, and handles any additional state variables defined
    within the neuron type.

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
    input_sig = model.sig[neurons]["in"] if input_sig is None else input_sig
    output_sig = model.sig[neurons]["out"] if output_sig is None else output_sig

    dtype = input_sig.dtype
    n_neurons = neurons.size_in
    rng = np.random.RandomState(model.seeds[neurons.ensemble] + 1)
    state_init = neurontype.make_state(n_neurons, rng=rng, dtype=dtype)
    state = {}

    for key, init in state_init.items():
        if key in model.sig[neurons]:
            raise BuildError("State name %r overlaps with existing signal name" % key)
        if is_array_like(init):
            model.sig[neurons][key] = Signal(
                initial_value=init, name="%s.%s" % (neurons, key)
            )
            state[key] = model.sig[neurons][key]
        elif isinstance(init, np.random.RandomState):
            # Pass through RandomState instances
            state[key] = init
        else:
            raise BuildError(
                "State %r is of type %r. Only array-likes and RandomStates are "
                "currently supported." % (key, type(init).__name__)
            )

    model.add_op(
        SimNeurons(neurons=neurontype, J=input_sig, output=output_sig, state=state)
    )


@Builder.register(RatesToSpikesNeuronType)
def build_rates_to_spikes(model, neurontype, neurons):
    """Builds a `.RatesToSpikesNeuronType` object into a model.

    This function adds two `.SimNeurons` operators. The first one handles
    simulating the base_type, converting input signals into rates. The second
    one takes those rates as input and emits spikes.

    Parameters
    ----------
    model : Model
        The model to build into.
    neurontype : RatesToSpikesNeuronType
        Neuron type to build.
    neuron : Neurons
        The neuron population object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.NeuronType` instance.
    """

    in_sig = model.sig[neurons]["in"]
    out_sig = model.sig[neurons]["out"]
    rate_sig = Signal(
        shape=model.sig[neurons]["in"].shape, name="%s.rate_out" % (neurons,),
    )
    model.sig[neurons]["rate_out"] = rate_sig

    # build the base neuron type
    build_neurons(
        model, neurontype.base_type, neurons, input_sig=in_sig, output_sig=rate_sig
    )

    # apply the RatesToSpikes conversion to output of base neuron type
    build_neurons(model, neurontype, neurons, input_sig=rate_sig, output_sig=out_sig)
