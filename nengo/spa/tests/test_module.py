import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.exceptions import SpaModuleError
from nengo.spa.utils import similarity
from nengo.spa.vocab import VocabularyMap


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


def test_spa_verification(seed, plt):
    d = 16

    model = spa.Module(seed=seed)

    # building a normal model that shouldn't raise a warning
    with model:
        model.buf = spa.Buffer(d)
        model.input_node = spa.Input(buf='B')
        # make sure errors aren't fired for non-spa modules
        prod = nengo.networks.Product(10, 2)  # noqa: F841
        model.int_val = 1

        # reassignment is fine for non-modules
        model.int_val = 2

    # reassignment of modules should throw an error
    with pytest.raises(ValueError):
        with model:
            model.buf = spa.State(d, feedback=1)

    with pytest.raises(ValueError):
        model = spa.Module(seed=seed)
        # build a model that should raise an error because no variable
        with model:
            model.buf = spa.Buffer(d)
            spa.Input(buf='B')

    with pytest.raises(ValueError):
        model = spa.Module(seed=seed)
        # build a model that should raise an error because no input attribute
        with model:
            model.buf = spa.Buffer(d)
            input_node = spa.Input(buf='B')
            input_node.label = "woop"

    with pytest.raises(SpaModuleError):
        model = spa.Module(seed=seed)
        # build a model that should raise an error because no buf attribute
        with model:
            buf = spa.Buffer(d)
            model.input_node = spa.Input(buf='B')
            buf.label = "woop"


def test_spa_module_exception():
    class MyException(Exception):
        pass

    class TestModule(spa.module.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            raise MyException()

    with pytest.raises(MyException):
        with spa.Module() as model:
            model.test = TestModule()


def test_spa_get():
    D = 16
    model = spa.Module()
    with model:
        model.buf1 = spa.State(D)
        model.buf2 = spa.State(D)
        model.compare = spa.Compare(D)

    assert model.get_module('buf1') is model.buf1
    assert model.get_module('buf1.default') is model.buf1
    assert model.get_module('buf2') is model.buf2
    assert model.get_module_input('buf1')[0] is model.buf1.input
    assert model.get_module_input('buf1.default')[0] is model.buf1.input
    assert model.get_module_output('buf1')[0] is model.buf1.output
    assert model.get_module_output('buf1.default')[0] is model.buf1.output
    assert model.get_module_input('compare.A')[0] is model.compare.inputA
    assert model.get_module_input('compare.B')[0] is model.compare.inputB

    with pytest.raises(SpaModuleError):
        model.get_module('dummy')
    with pytest.raises(SpaModuleError):
        model.get_module_input('dummy')
    with pytest.raises(SpaModuleError):
        model.get_module_output('dummy')
    with pytest.raises(SpaModuleError):
        model.get_module_input('buf1.A')
    with pytest.raises(SpaModuleError):
        model.get_module_input('compare')


def test_spa_vocab():
    # create a model without a vocab and check that it is empty
    model = spa.Module()
    assert len(model.vocabs) == 0

    # create a model with a vocab and check that it's filled
    va = spa.Vocabulary(16)
    va.parse("PANTS")
    vb = spa.Vocabulary(32)
    vb.parse("SHOES")
    model = spa.Module(vocabs=VocabularyMap([va, vb]))
    assert model.vocabs[16].keys == ["PANTS"]
    assert model.vocabs[32].keys == ["SHOES"]

    # warning on vocabs with duplicate dimensions
    vc = spa.Vocabulary(16)
    vc.parse("SOCKS")
    with pytest.warns(UserWarning):
        model = spa.Module(vocabs=VocabularyMap([va, vb, vc]))
    assert model.vocabs[16].keys == ["SOCKS"]
    assert model.vocabs[32].keys == ["SHOES"]


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
