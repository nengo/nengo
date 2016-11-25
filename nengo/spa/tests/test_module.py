import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.exceptions import SpaModuleError, SpaTypeError
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

            spa.Actions('state_out = state_in').build()

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
        model.buf = spa.State(d)
        model.input_node = spa.Input()
        model.input_node.buf = 'B'
        # make sure errors aren't fired for non-spa modules
        prod = nengo.networks.Product(10, 2)  # noqa: F841
        model.int_val = 1

        # reassignment is fine for non-modules
        model.int_val = 2

    # reassignment of modules should throw an error
    with pytest.raises(ValueError):
        with model:
            model.buf = spa.State(d, feedback=1)


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
    with pytest.raises(SpaModuleError):
        model.get_module('buf1.default')
    assert model.get_module('buf2') is model.buf2
    assert model.get_module_input('buf1')[0] is model.buf1.input
    assert model.get_module_input('buf1.default')[0] is model.buf1.input
    assert model.get_module_output('buf1')[0] is model.buf1.output
    assert model.get_module_output('buf1.default')[0] is model.buf1.output
    assert model.get_module_input(
        'compare.input_a')[0] is model.compare.input_a
    assert model.get_module_input(
        'compare.input_b')[0] is model.compare.input_b

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
    va.populate("PANTS")
    vb = spa.Vocabulary(32)
    vb.populate("SHOES")
    model = spa.Module(vocabs=VocabularyMap([va, vb]))
    assert list(model.vocabs[16].keys()) == ["PANTS"]
    assert list(model.vocabs[32].keys()) == ["SHOES"]

    # warning on vocabs with duplicate dimensions
    vc = spa.Vocabulary(16)
    vc.populate("SOCKS")
    with pytest.warns(UserWarning):
        model = spa.Module(vocabs=VocabularyMap([va, vb, vc]))
    assert list(model.vocabs[16].keys()) == ["SOCKS"]
    assert list(model.vocabs[32].keys()) == ["SHOES"]


def test_hierarchical(Simulator, seed, plt):
    d = 32

    with spa.Module(seed=seed) as model:
        model.comm_channel = SpaCommunicationChannel(d)
        model.out = spa.State(d)

        spa.Actions('out = comm_channel').build()
        model.stimulus = spa.Input()
        model.stimulus.comm_channel = 'A'

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

        spa.Actions('out = comm_channel.state_out').build()
        model.stimulus = spa.Input()
        model.stimulus.comm_channel.state_in = 'A'

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


def test_no_magic_vocab_transform():
    d = 16
    v1 = spa.Vocabulary(d)
    v2 = spa.Vocabulary(d)

    with spa.Module() as model:
        model.a = spa.State(vocab=v1)
        model.b = spa.State(vocab=v2)
        with pytest.raises(SpaTypeError):
            spa.Actions('b = a').build()


@pytest.mark.parametrize('d1,d2,method,lookup', [
    (16, 16, 'reinterpret(a, v2)', 'v1'),
    (16, 16, 'reinterpret(a)', 'v1'),
    (16, 16, 'reinterpret(a, b)', 'v1'),
    (16, 32, 'translate(a, v2)', 'v2'),
    (16, 32, 'translate(a)', 'v2'),
    (16, 32, 'translate(a, b)', 'v2')])
def test_casting_vocabs(d1, d2, method, lookup, Simulator, plt, rng):
    v1 = spa.Vocabulary(d1, rng=rng)
    v1.populate('A')
    v2 = spa.Vocabulary(d2, rng=rng)
    v2.populate('A')

    with spa.Module() as model:
        model.a = spa.State(vocab=v1)
        model.b = spa.State(vocab=v2)
        spa.Actions('b = {}'.format(method), vocabs={'v2': v2}).build()
        model.stimulus = spa.Input()
        model.stimulus.a = 'A'
        p = nengo.Probe(model.b.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.5)

    t = sim.trange() > 0.2
    v = locals()[lookup].parse('A').v

    plt.plot(sim.trange(), similarity(sim.data[p], v))
    plt.xlabel("t [s]")
    plt.ylabel("Similarity")

    assert np.mean(similarity(sim.data[p][t], v)) > 0.8
