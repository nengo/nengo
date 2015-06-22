import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Operator
from nengo.neurons import (AdaptiveLIF, AdaptiveLIFRate, Izhikevich, LIF,
                           LIFRate, RectifiedLinear, Sigmoid)


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

        def step_simneurons():
            self.neurons.step_math(dt, J, output, *states)
        return step_simneurons


@Builder.register(RectifiedLinear)
def build_rectifiedlinear(model, reclinear, neurons):
    model.add_op(SimNeurons(neurons=reclinear,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(Sigmoid)
def build_sigmoid(model, sigmoid, neurons):
    model.add_op(SimNeurons(neurons=sigmoid,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIFRate)
def build_lifrate(model, lifrate, neurons):
    model.add_op(SimNeurons(neurons=lifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out']))


@Builder.register(LIF)
def build_lif(model, lif, neurons):
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
def build_alifrate(model, alifrate, neurons):
    model.sig[neurons]['adaptation'] = Signal(
        np.zeros(neurons.size_in), name="%s.adaptation" % neurons)
    model.add_op(SimNeurons(neurons=alifrate,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['adaptation']]))


@Builder.register(AdaptiveLIF)
def build_alif(model, alif, neurons):
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


@Builder.register(Izhikevich)
def build_izhikevich(model, izhikevich, neurons):
    model.sig[neurons]['voltage'] = Signal(
        np.ones(neurons.size_in) * izhikevich.reset_voltage,
        name="%s.voltage" % neurons)
    model.sig[neurons]['recovery'] = Signal(
        np.ones(neurons.size_in)
        * izhikevich.reset_voltage
        * izhikevich.coupling, name="%s.recovery" % neurons)
    model.add_op(SimNeurons(neurons=izhikevich,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['recovery']]))
