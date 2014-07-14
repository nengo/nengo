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

        self.reads = [J]
        self.updates = [output] + states
        self.sets = []
        self.incs = []

    def make_step(self, signals, dt):
        J = signals[self.J]
        output = signals[self.output]
        states = [signals[state] for state in self.states]

        def step():
            self.neurons.step_math(dt, J, output, *states)
        return step


@Builder.register_builder(LIFRate)
def build_lifrate(lif, ens, model, config):
    model.add_op(SimNeurons(neurons=lif,
                            J=model.sig[ens]['neuron_in'],
                            output=model.sig[ens]['neuron_out']))


@Builder.register_builder(LIF)
def build_lif(lif, ens, model, config):
    model.sig[ens]['voltage'] = Signal(
        np.zeros(ens.n_neurons), name="%s.voltage" % ens.label)
    model.sig[ens]['refractory_time'] = Signal(
        np.zeros(ens.n_neurons), name="%s.refractory_time" % ens.label)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[ens]['neuron_in'],
        output=model.sig[ens]['neuron_out'],
        states=[model.sig[ens]['voltage'], model.sig[ens]['refractory_time']]))


@Builder.register_builder(AdaptiveLIFRate)
def build_alifrate(alif, ens, model, config):
    model.sig[ens]['adaptation'] = Signal(
        np.zeros(ens.n_neurons), name="%s.adaptation" % ens.label)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[ens]['neuron_in'],
                            output=model.sig[ens]['neuron_out'],
                            states=[model.sig[ens]['adaptation']]))


@Builder.register_builder(AdaptiveLIF)
def build_alif(alif, ens, model, config):
    model.sig[ens]['voltage'] = Signal(
        np.zeros(ens.n_neurons), name="%s.voltage" % ens.label)
    model.sig[ens]['refractory_time'] = Signal(
        np.zeros(ens.n_neurons), name="%s.refractory_time" % ens.label)
    model.sig[ens]['adaptation'] = Signal(
        np.zeros(ens.n_neurons), name="%s.adaptation" % ens.label)
    model.add_op(SimNeurons(neurons=alif,
                            J=model.sig[ens]['neuron_in'],
                            output=model.sig[ens]['neuron_out'],
                            states=[model.sig[ens]['voltage'],
                                    model.sig[ens]['refractory_time'],
                                    model.sig[ens]['adaptation']]))
