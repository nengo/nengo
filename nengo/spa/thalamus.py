import numpy as np

import nengo
from nengo.spa.module import Module
from nengo.utils.compat import iteritems


class Thalamus(Module):
    def __init__(self, bg, neurons_per_rule=50, inhibit=1, pstc_inhibit=0.008,
                 output_synapse=0.01, rule_threshold=0.2,
                 neurons_per_channel_dim=50, channel_subdim=16,
                 channel_pstc=0.01, neurons_cconv=200,
                 neurons_gate=40, gate_threshold=0.3, pstc_to_gate=0.002):
        super(Thalamus, self).__init__()
        self.bg = bg
        self.neurons_per_rule = neurons_per_rule
        self.inhibit = inhibit
        self.pstc_inhibit = pstc_inhibit
        self.output_synapse = output_synapse
        self.rule_threshold = rule_threshold
        self.neurons_per_channel_dim = neurons_per_channel_dim
        self.channel_subdim = channel_subdim
        self.channel_pstc = channel_pstc
        self.neurons_gate = neurons_gate
        self.neurons_cconv = neurons_cconv
        self.gate_threshold = gate_threshold
        self.pstc_to_gate = pstc_to_gate

    def on_add(self, spa):  # noqa: C901
        Module.on_add(self, spa)

        N = self.bg.rules.count

        rules = nengo.networks.EnsembleArray(
            self.neurons_per_rule,
            N, ens_dimensions=1,
            intercepts=nengo.objects.Uniform(self.rule_threshold, 1),
            label='rules')
        self.rules = rules

        for ens in rules.ensembles:
            ens.encoders = [[1.0]] * self.neurons_per_rule

        bias = nengo.Node(output=[1], label='bias')
        self.bias = bias

        nengo.Connection(rules.output, rules.input,
                         transform=(np.eye(N)-1)*self.inhibit,
                         synapse=self.pstc_inhibit)

        nengo.Connection(bias, rules.input, transform=np.ones((N, 1)),
                         synapse=None)

        nengo.Connection(self.bg.output, rules.input, synapse=None)

        for output, transform in iteritems(self.bg.rules.get_outputs_direct()):
            nengo.Connection(rules.output, output, transform=transform,
                             synapse=self.output_synapse)

        for index, route in self.bg.rules.get_outputs_route():
            target, source = route

            dim = target.vocab.dimensions

            gate = nengo.Ensemble(
                self.neurons_gate, dimensions=1,
                intercepts=nengo.objects.Uniform(self.gate_threshold, 1),
                label='gate_%d_%s' % (index, target.name))
            gate.encoders = [[1]] * self.neurons_gate

            nengo.Connection(rules.ensembles[index], gate, transform=-1,
                             synapse=self.pstc_to_gate)
            nengo.Connection(bias, gate, synapse=None)

            if hasattr(source, 'convolve'):
                # TODO: this is an insanely bizarre computation to have to do
                #   whenever you want to use a CircConv network. The parameter
                #   should be changed to specify neurons per ensemble
                n_neurons_d = self.neurons_cconv * (
                    2 * dim - (2 if dim % 2 == 0 else 1))
                channel = nengo.networks.CircularConvolution(
                    n_neurons_d, dim,
                    invert_a=source.invert, invert_b=source.convolve.invert,
                    label='cconv_%d_%s' % (index, target.name))

                nengo.Connection(channel.output, target.obj,
                                 synapse=self.channel_pstc)

                transform = [[-1]] * (self.neurons_cconv)
                for e in channel.ensemble.ensembles:
                    nengo.Connection(gate, e.neurons, transform=transform,
                                     synapse=self.pstc_inhibit)

                # connect first input
                if target.vocab is source.vocab:
                    transform = 1
                else:
                    transform = source.vocab.transform_to(target.vocab)

                if hasattr(source, 'transform'):
                    t2 = source.vocab.parse(source.transform)
                    t2 = t2.get_convolution_matrix()
                    transform = np.dot(transform, t2)

                nengo.Connection(source.obj, channel.A, transform=transform,
                                 synapse=self.channel_pstc)

                # connect second input
                if target.vocab is source.convolve.vocab:
                    transform = 1
                else:
                    transform = source.convolve.vocab
                    transform = transform.transform_to(target.vocab)

                if hasattr(source.convolve, 'transform'):
                    vocab = source.convolve.vocab
                    t2 = vocab.parse(source.convolve.transform)
                    t2 = t2.get_convolution_matrix()
                    transform = np.dot(transform, t2)

                nengo.Connection(source.convolve.obj, channel.B,
                                 transform=transform,
                                 synapse=self.channel_pstc)

            else:

                if source.invert:
                    raise Exception('Inverting on a communication channel '
                                    'not supported yet')

                subdim = self.channel_subdim
                assert dim % subdim == 0  # TODO: Add these asserts elsewhere
                channel = nengo.networks.EnsembleArray(
                    self.neurons_per_channel_dim * subdim,
                    dim // subdim, ens_dimensions=subdim,
                    label='channel_%d_%s' % (index, target.name))

                nengo.Connection(channel.output, target.obj,
                                 synapse=self.channel_pstc)

                transform = [[-1]] * (self.neurons_per_channel_dim * subdim)
                for e in channel.ensembles:
                    nengo.Connection(gate, e.neurons, transform=transform,
                                     synapse=self.pstc_inhibit)

                if target.vocab is source.vocab:
                    transform = 1
                else:
                    transform = source.vocab.transform_to(target.vocab)

                if hasattr(source, 'transform'):
                    t2 = source.vocab.parse(source.transform)
                    t2 = t2.get_convolution_matrix()
                    transform = np.dot(transform, t2)

                nengo.Connection(source.obj, channel.input,
                                 transform=transform,
                                 synapse=self.channel_pstc)
