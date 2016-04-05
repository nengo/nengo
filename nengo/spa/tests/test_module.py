import numpy as np

import nengo
from nengo import spa
from nengo.spa.utils import similarity


class SpaCommunicationChannel(spa.Module):
    def __init__(
            self, dimensions, label=None, seed=None, add_to_container=None):
        super(SpaCommunicationChannel, self).__init__(
            label, seed, add_to_container)

        with self:
            self.state_in = spa.State(dimensions)
            self.state_out = spa.State(dimensions)

            self.cortical = spa.Cortical(spa.Actions('state_out = state_in'))

        self.inputs = dict(
            default=self.state_in.inputs['default'],
            secondary=self.state_in.inputs['default'])
        self.outputs = dict(
            default=self.state_out.outputs['default'],
            secondary=self.state_out.outputs['default'])


def test_hierarchical(Simulator, seed, plt):
    d = 32

    with spa.Module(seed=seed) as model:
        model.comm_channel = SpaCommunicationChannel(d)
        model.out = spa.State(d)

        model.cortical = spa.Cortical(spa.Actions('out = comm_channel'))
        model.stimulus = spa.Input(comm_channel='A')

        p = nengo.Probe(model.out.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    t = sim.trange() > 0.2
    v = model.vocabs[d].parse('A').v

    plt.plot(sim.trange(), similarity(sim.data[p], v))
    plt.xlabel("t [s]")
    plt.ylabel("Similarity")

    assert np.mean(similarity(sim.data[p][t], v)) > 0.8


def test_hierarichal_module_name_resolution():
    with spa.Module() as model:
        model.comm_channel = SpaCommunicationChannel(16)

    assert (
        model.get_module('comm_channel.state_in') is
        model.comm_channel.state_in)
    assert (
        model.get_module_input('comm_channel.state_in') is
        model.comm_channel.state_in.inputs['default'])
    assert (
        model.get_module_input('comm_channel.secondary') is
        model.comm_channel.inputs['secondary'])
    assert (
        model.get_module_output('comm_channel.state_out') is
        model.comm_channel.state_out.outputs['default'])
    assert (
        model.get_module_output('comm_channel.secondary') is
        model.comm_channel.outputs['secondary'])


def test_hierarchical_actions(Simulator, seed, plt):
    d = 32

    with spa.Module(seed=seed) as model:
        model.comm_channel = SpaCommunicationChannel(d)
        model.out = spa.State(d)

        model.cortical = spa.Cortical(
            spa.Actions('out = comm_channel.state_out'))
        model.stimulus = spa.Input(**{'comm_channel.state_in': 'A'})

        p = nengo.Probe(model.out.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    t = sim.trange() > 0.2
    v = model.vocabs[d].parse('A').v

    plt.plot(sim.trange(), similarity(sim.data[p], v))
    plt.xlabel("t [s]")
    plt.ylabel("Similarity")

    assert np.mean(similarity(sim.data[p][t], v)) > 0.8


def test_vocab_config():
    with spa.Module() as model:
        with spa.Module() as model.shared_vocabs:
            pass
        with spa.Module(vocabs=spa.VocabularyMap()) as model.non_shared_vocabs:
            pass

    assert model.shared_vocabs.vocabs is model.vocabs
    assert model.non_shared_vocabs.vocabs is not model.vocabs
