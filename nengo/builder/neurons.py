import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.neurons import LIF, LIFRate, AdaptiveLIF, AdaptiveLIFRate


class SimNeurons(Operator):
    """Set output to neuron model output for the given input current."""

    def __init__(self, neurons, J, output, states=[]):
        self.neurons = neurons
        self.J = J
        self.output = output
        self.states = states

        self.sets = [output] + states
        self.incs = []
        self.reads = [J]
        self.updates = []

    def make_step(self, signals, dt, rng):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        def step():
            self.neurons.step_math(dt, J, output, *states)
        return step


@Builder.register(LIFRate)
def build_lifrate(lifrate, neurons, model, config):
    model.add_op(SimNeurons(neurons=lifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIF)
def build_lif(lif, neurons, model, config):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time']]))


@Builder.register(AdaptiveLIFRate)
def build_alifrate(alifrate, neurons, model, config):
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['adaptation']]))


@Builder.register(AdaptiveLIF)
def build_alif(alif, neurons, model, config):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['refractory_time'],
                                    model.sig[neurons]['adaptation']]))
