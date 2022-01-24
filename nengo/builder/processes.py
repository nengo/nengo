import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.processes import Process
from nengo.rc import rc
from nengo.synapses import Synapse


class SimProcess(Operator):
    """Simulate a process.

    Parameters
    ----------
    process : Process
        The `.Process` to simulate.
    input : Signal or None
        Input to the process, or None if no input.
    output : Signal or None
        Output from the process, or None if no output.
    t : Signal
        The signal associated with the time (a float, in seconds).
    mode : str, optional
        Denotes what type of update this operator performs.
        Must be one of ``'update'``, ``'inc'`` or ``'set'``.
    tag : str, optional
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    input : Signal or None
        Input to the process, or None if no input.
    mode : str
        Denotes what type of update this operator performs.
    output : Signal or None
        Output from the process, or None if no output.
    process : Process
        The `.Process` to simulate.
    t : Signal
        The signal associated with the time (a float, in seconds).
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[output] if output is not None and mode=='set' else []``
    2. incs ``[output] if output is not None and mode=='inc' else []``
    3. reads ``[t, input] if input is not None else [t]``
    4. updates ``[output] if output is not None and mode=='update' else []``
    """

    def __init__(self, process, input, output, t, mode="set", state=None, tag=None):
        super().__init__(tag=tag)
        self.process = process
        self.mode = mode

        assert output is not None

        self.reads = [t, input] if input is not None else [t]
        self.sets = []
        self.incs = []
        self.updates = []
        if mode == "update":
            self.updates.extend([output])
        elif mode == "inc":
            self.incs.extend([output])
        elif mode == "set":
            self.sets.extend([output])
        else:
            raise ValueError(f"Unrecognized mode '{mode}'")

        self.state_idxs = {}
        if state is not None:
            for name, sig in state.items():
                # The signals actually stored in `self.updates` can be modified by the
                # optimizer. To allow this possibility, we store the index of the
                # signal in the updates list instead of storing the signal itself.
                self.state_idxs[name] = len(self.updates)
                self.updates.append(sig)

    @property
    def input(self):
        return None if len(self.reads) == 1 else self.reads[1]

    @property
    def output(self):
        if self.mode == "update":
            return self.updates[0]
        if self.mode == "inc":
            return self.incs[0]
        assert self.mode == "set"
        return self.sets[0]

    @property
    def state(self):
        return {key: self.updates[idx] for key, idx in self.state_idxs.items()}

    @property
    def t(self):
        return self.reads[0]

    @property
    def _descstr(self):
        return f"{self.process}, {self.input} -> {self.output}"

    def make_step(self, signals, dt, rng):
        t = signals[self.t]
        input = signals[self.input] if self.input is not None else None
        output = signals[self.output]
        shape_in = input.shape if input is not None else (0,)
        shape_out = output.shape
        rng = self.process.get_rng(rng)
        state = {name: signals[sig] for name, sig in self.state.items()}
        step_f = self.process.make_step(shape_in, shape_out, dt, rng, state)

        if self.mode == "inc" and input is None:

            def step_simprocess():
                output[...] += step_f(t.item())

        elif self.mode == "inc":

            def step_simprocess():
                output[...] += step_f(t.item(), np.copy(input))

        elif input is None:

            def step_simprocess():
                output[...] = step_f(t.item())

        else:
            assert self.mode != "inc" and input is not None

            def step_simprocess():
                output[...] = step_f(t.item(), np.copy(input))

        return step_simprocess


@Builder.register(Process)
def build_process(
    model, process, sig_in=None, sig_out=None, mode="set", seed_or_rng=0, **kwargs
):
    """Builds a `.Process` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    process : Process
        Process to build.
    sig_in : Signal, optional
        The input signal, or None if no input signal.
    sig_out : Signal, optional
        The output signal, or None if no output signal.
    mode : "set" or "inc" or "update", optional
        The ``mode`` of the built `.SimProcess`.
    seed_or_rng : int or `numpy.random.RandomState`
        The parent seed or random number generator to use if the seed is not set.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Process` instance.
    """
    shape_in = sig_in.shape if sig_in is not None else (0,)
    shape_out = sig_out.shape if sig_out is not None else (0,)
    dtype = (
        sig_out.dtype
        if sig_out is not None
        else sig_in.dtype
        if sig_in is not None
        else rc.float_dtype
    )
    state_rng = process.get_rng(seed_or_rng, offset=1)  # offset matches `Process.apply`
    state_init = process.make_state(
        shape_in, shape_out, model.dt, rng=state_rng, dtype=dtype, **kwargs
    )
    state = {}
    for name, value in state_init.items():
        state[name] = Signal(value)
        model.sig[process]["_state_" + name] = state[name]

    model.add_op(
        SimProcess(
            process,
            sig_in,
            sig_out,
            model.time,
            mode=mode,
            state=state,
            tag=str(process),
        )
    )

    return sig_out


@Builder.register(Synapse)
def build_synapse(model, synapse, sig_in=None, sig_out=None, mode="set", seed_or_rng=0):
    """Builds a `.Synapse` object into a model.

    Wrapper around `.build_process` that configures the output signal with the
    initial output.

    Parameters
    ----------
    model : Model
        The model to build into.
    synapse : Synapse
        Synapse to build.
    sig_in : Signal, optional
        The input signal, or None if no input signal.
    sig_out : Signal, optional
        The output signal, or None if no output signal.
    mode : "set" or "inc" or "update", optional
        The ``mode`` of the built `.SimProcess`.
    seed_or_rng : int or `numpy.random.RandomState`
        The parent seed or random number generator to use if the seed is not set.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Synapse` instance.
    """
    y0 = None
    if sig_out is None:
        assert sig_in is not None, "Both `sig_in` and `sig_out` cannot be None"
        output_shape = sig_in.shape

        # state_rng offset matches `Process.apply`
        state_rng = synapse.get_rng(seed_or_rng, offset=1)
        y0 = synapse._sample_initial_output(output_shape, rng=state_rng)
        if y0 is None:
            y0 = np.zeros(output_shape, dtype=rc.float_dtype)
        else:
            y0 = y0 * np.ones(output_shape, dtype=rc.float_dtype)  # broadcast

        assert y0.shape == output_shape
        sig_out = Signal(y0, name=f"{sig_in.name}.{synapse}")

    return build_process(
        model,
        synapse,
        sig_in=sig_in,
        sig_out=sig_out,
        mode=mode,
        seed_or_rng=seed_or_rng,
        y0=y0,
    )
