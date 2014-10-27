import collections

import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.synapses import Alpha, LinearFilter, Lowpass, Synapse
from nengo.utils.compat import is_number
from nengo.utils.filter_design import cont2discrete


class SimFilterSynapse(Operator):
    """Simulate a discrete-time LTI system.

    Implements a discrete-time LTI system using the difference equation [1]_
    for the given transfer function (num, den).

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Digital_filter#Difference_equation
    """
    def __init__(self, input, output, num, den):
        self.input = input
        self.output = output
        self.num = num
        self.den = den

        self.sets = []
        self.incs = []
        self.reads = [input]
        self.updates = [output]

    def make_step(self, signals, dt, rng):
        input = signals[self.input]
        output = signals[self.output]
        num, den = self.num, self.den

        if len(num) == 1 and len(den) == 0:
            def step(input=input, output=output, b=num[0]):
                output[:] = b * input
        elif len(num) == 1 and len(den) == 1:
            def step(input=input, output=output, a=den[0], b=num[0]):
                output *= -a
                output += b * input
        else:
            x = collections.deque(maxlen=len(num))
            y = collections.deque(maxlen=len(den))

            def step(input=input, output=output, x=x, y=y, num=num, den=den):
                output[:] = 0

                x.appendleft(np.array(input))
                for k, xk in enumerate(x):
                    output += num[k] * xk
                for k, yk in enumerate(y):
                    output -= den[k] * yk
                y.appendleft(np.array(output))

        return step


def filtered_signal(model, owner, sig, synapse):
    # Note: we add a filter here even if synapse < dt,
    # in order to avoid cycles in the op graph. If the filter
    # is explicitly set to None (e.g. for a passthrough node)
    # then cycles can still occur.
    if is_number(synapse):
        synapse = Lowpass(synapse)
    assert isinstance(synapse, Synapse)
    model.build(synapse, owner, sig)
    return model.sig[owner]['synapse_out']


def build_discrete_filter(model, synapse, owner, input_signal, num, den):
    model.sig[owner]['synapse_in'] = input_signal
    model.sig[owner]['synapse_out'] = Signal(
        np.zeros(input_signal.size),
        name="%s.%s" % (input_signal.name, synapse))

    model.add_op(SimFilterSynapse(input=model.sig[owner]['synapse_in'],
                                  output=model.sig[owner]['synapse_out'],
                                  num=num, den=den))


@Builder.register(LinearFilter)
def build_filter(model, synapse, owner, input_signal):
    num, den, _ = cont2discrete(
        (synapse.num, synapse.den), model.dt, method='zoh')
    num = num.flatten()
    num = num[1:] if num[0] == 0 else num
    den = den[1:]  # drop first element (equal to 1)
    build_discrete_filter(model, synapse, owner, input_signal, num, den)


@Builder.register(Lowpass)
def build_lowpass(model, synapse, owner, input_signal):
    if synapse.tau > 0.03 * model.dt:
        d = -np.expm1(-model.dt / synapse.tau)
        num, den = [d], [d - 1]
    else:
        num, den = [1.], []

    build_discrete_filter(model, synapse, owner, input_signal, num, den)


@Builder.register(Alpha)
def build_alpha(model, synapse, owner, input_signal):
    if synapse.tau > 0.03 * model.dt:
        a = model.dt / synapse.tau
        ea = np.exp(-a)
        num, den = [-a*ea + (1 - ea), ea*(a + ea - 1)], [-2 * ea, ea**2]
    else:
        num, den = [1.], []  # just copy the input

    build_discrete_filter(model, synapse, owner, input_signal, num, den)
