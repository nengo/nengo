from copy import copy
import pickle

import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.exceptions import NetworkContextError, NotAddedToNetworkWarning
from nengo.params import IntParam, iter_params
from nengo.utils.numpy import is_array_like


def assert_is_copy(cp, original):
    assert cp is not original  # ensures separate parameters
    for param in iter_params(cp):
        param_inst = getattr(cp, param)
        if isinstance(param_inst, nengo.solvers.Solver) or isinstance(
                param_inst, nengo.base.NengoObject):
            assert param_inst is getattr(original, param)
        elif is_array_like(param_inst):
            assert np.all(param_inst == getattr(original, param))
        else:
            assert param_inst == getattr(original, param)


def assert_is_deepcopy(cp, original):
    assert cp is not original  # ensures separate parameters
    for param in iter_params(cp):
        param_inst = getattr(cp, param)
        if isinstance(param_inst, nengo.solvers.Solver) or isinstance(
                param_inst, nengo.base.NengoObject):
            assert_is_copy(param_inst, getattr(original, param))
        elif is_array_like(param_inst):
            assert np.all(param_inst == getattr(original, param))
        else:
            assert param_inst == getattr(original, param)


def make_ensemble():
    with nengo.Network():
        e = nengo.Ensemble(10, 1, radius=2.)
    return e


def make_probe():
    with nengo.Network():
        e = nengo.Ensemble(10, 1)
        p = nengo.Probe(e, synapse=0.01)
    return p


def make_node():
    with nengo.Network():
        n = nengo.Node(np.min, size_in=2, size_out=2)
    return n


def make_connection():
    with nengo.Network():
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        c = nengo.Connection(e1, e2, transform=2.)
    return c


def make_function_connection():
    with nengo.Network():
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        c = nengo.Connection(e1, e2, function=lambda x: x**2)
    return c


def make_learning_connection():
    """Test pickling LearningRule and Neurons"""
    with nengo.Network():
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(11, 1)
        c = nengo.Connection(e1.neurons, e2.neurons,
                             transform=np.ones((11, 10)))
        c.learning_rule_type = nengo.PES()
        nengo.Connection(e2, c.learning_rule)
    return c


def make_network():
    with nengo.Network() as model:
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e1, e2, transform=2.)
        nengo.Probe(e2)
    return model


def test_neurons_reference_copy():
    original = make_ensemble()
    cp = original.copy(add_to_container=False)
    assert original.neurons.ensemble is original
    assert cp.neurons.ensemble is cp


def test_learningrule_reference_copy():
    original = make_learning_connection()
    cp = original.copy(add_to_container=False)
    assert original.learning_rule.connection is original
    assert cp.learning_rule.connection is cp


def test_copy_in_network_default_add():
    original = make_network()

    with nengo.Network() as model:
        cp = original.copy()
    assert cp in model.all_objects

    assert_is_deepcopy(cp, original)


def test_copy_outside_network_default_add():
    original = make_network()
    cp = original.copy()
    assert_is_deepcopy(cp, original)


def test_network_copies_defaults():
    original = nengo.Network()
    original.config[nengo.Ensemble].radius = 1.5
    original.config[nengo.Connection].synapse = nengo.Lowpass(0.1)

    cp = original.copy()
    assert cp.config[nengo.Ensemble].radius == 1.5
    assert cp.config[nengo.Connection].synapse == nengo.Lowpass(0.1)


def test_network_copy_allows_independent_manipulation():
    with nengo.Network() as original:
        nengo.Ensemble(10, 1)
    original.config[nengo.Ensemble].radius = 2.

    cp = original.copy()
    with cp:
        e2 = nengo.Ensemble(10, 1)
    cp.config[nengo.Ensemble].radius = 1.

    assert e2 not in original.ensembles
    assert original.config[nengo.Ensemble].radius == 2.


def test_copies_structure():
    with nengo.Network() as original:
        e1 = nengo.Ensemble(10, 1)
        e2 = nengo.Ensemble(10, 1)
        nengo.Connection(e1, e2)
        nengo.Probe(e2)

    cp = original.copy()

    assert cp.connections[0].pre is not e1
    assert cp.connections[0].post is not e2
    assert cp.connections[0].pre in cp.ensembles
    assert cp.connections[0].post in cp.ensembles

    assert cp.probes[0].target is not e2
    assert cp.probes[0].target in cp.ensembles


def test_network_copy_builds(RefSimulator):
    with RefSimulator(make_network().copy()):
        pass


def test_copy_obj_view():
    with nengo.Network():
        ens = nengo.Ensemble(10, 4)
        original = ens[:2]

    cp = original.copy()

    assert cp is not original
    assert cp.obj is ens
    assert cp.slice == original.slice


def test_copy_obj_view_in_connection():
    with nengo.Network() as original:
        node = nengo.Node([0.1, 0.2])
        ens = nengo.Ensemble(10, 2)
        nengo.Connection(node[0], ens[1])
        nengo.Connection(node[1], ens[0])

    cp = original.copy()
    assert cp.nodes[0] is not node
    assert cp.ensembles[0] is not ens
    assert cp.connections[0].pre.obj is cp.nodes[0]
    assert cp.connections[1].pre.obj is cp.nodes[0]
    assert cp.connections[0].post.obj is cp.ensembles[0]
    assert cp.connections[1].post.obj is cp.ensembles[0]


def test_pickle_obj_view():
    with nengo.Network():
        ens = nengo.Ensemble(10, 4)
        original = ens[:2]

    cp = pickle.loads(pickle.dumps(original))

    assert cp is not original
    assert cp.obj is not ens
    assert cp.obj.n_neurons == 10
    assert cp.obj.dimensions == 4
    assert cp.slice == original.slice


def test_pickle_obj_view_in_connection():
    with nengo.Network() as original:
        node = nengo.Node([0.1, 0.2])
        ens = nengo.Ensemble(10, 2)
        nengo.Connection(node[0], ens[1])
        nengo.Connection(node[1], ens[0])

    cp = pickle.loads(pickle.dumps(original))
    assert cp.nodes[0] is not node
    assert cp.ensembles[0] is not ens
    assert cp.connections[0].pre.obj is cp.nodes[0]
    assert cp.connections[1].pre.obj is cp.nodes[0]
    assert cp.connections[0].post.obj is cp.ensembles[0]
    assert cp.connections[1].post.obj is cp.ensembles[0]


@pytest.mark.parametrize(('make_f', 'assert_f'), [
    (make_ensemble, assert_is_copy),
    (make_probe, assert_is_copy),
    (make_node, assert_is_copy),
    (make_connection, assert_is_copy),
    (make_function_connection, assert_is_copy),
    (make_learning_connection, assert_is_copy),
    (make_network, assert_is_copy),
])
class TestCopy:
    """A basic set of tests that should pass for all objects."""

    def test_copy_in_network(self, make_f, assert_f):
        original = make_f()

        with nengo.Network() as model:
            cp = original.copy(add_to_container=True)
        assert cp in model.all_objects

        assert_f(cp, original)

    def test_copy_in_network_without_adding(self, make_f, assert_f):
        original = make_f()

        with nengo.Network() as model:
            cp = original.copy(add_to_container=False)
        assert cp not in model.all_objects

        assert_f(cp, original)

    def test_copy_outside_network(self, make_f, assert_f):
        original = make_f()
        with pytest.raises(NetworkContextError):
            original.copy(add_to_container=True)

    def test_copy_outside_network_without_adding(self, make_f, assert_f):
        original = make_f()
        cp = original.copy(add_to_container=False)
        assert_f(cp, original)

    def test_python_copy_warns_abt_adding_to_network(self, make_f, assert_f):
        original = make_f()
        copy(original)  # Fine because not in a network
        with nengo.Network():
            with pytest.warns(NotAddedToNetworkWarning):
                copy(original)


@pytest.mark.parametrize('make_f', (
    make_ensemble, make_probe, make_node, make_connection, make_network
))
class TestPickle:
    """A basic set of tests that should pass for all objects."""

    def test_pickle_roundtrip(self, make_f):
        original = make_f()
        cp = pickle.loads(pickle.dumps(original))
        assert_is_deepcopy(cp, original)

    def test_unpickling_warning_in_network(self, make_f):
        original = make_f()
        pkl = pickle.dumps(original)
        with nengo.Network():
            with pytest.warns(NotAddedToNetworkWarning):
                pickle.loads(pkl)


@pytest.mark.parametrize('original', [
    nengo.learning_rules.PES(),
    nengo.learning_rules.BCM(),
    nengo.learning_rules.Oja(),
    nengo.learning_rules.Voja(),
    nengo.processes.WhiteNoise(),
    nengo.processes.FilteredNoise(),
    nengo.processes.BrownNoise(),
    nengo.processes.PresentInput([.1, .2], 1.),
    nengo.synapses.LinearFilter([.1, .2], [.3, .4], True),
    nengo.synapses.Lowpass(0.005),
    nengo.synapses.Alpha(0.005),
    nengo.synapses.Triangle(0.005),
])
class TestFrozenObjectCopies:

    def test_copy(self, original):
        assert_is_copy(copy(original), original)

    def test_pickle_roundtrip(self, original):
        assert_is_deepcopy(pickle.loads(pickle.dumps(original)), original)


def test_copy_spa(RefSimulator):
    with spa.SPA() as original:
        original.state = spa.State(16)
        original.cortex = spa.Cortical(spa.Actions("state = A"))

    cp = original.copy()

    # Check that it still builds.
    with RefSimulator(cp):
        pass

    # check vocab instance param is set
    for node in cp.all_nodes:
        if node.label in ['input', 'output']:
            assert cp.config[node].vocab is not None


def test_copy_instance_params():
    with nengo.Network() as original:
        original.config[nengo.Ensemble].set_param(
            'test', IntParam('test', optional=True))
        ens = nengo.Ensemble(10, 1)
        original.config[ens].test = 42

    cp = original.copy()
    assert cp.config[cp.ensembles[0]].test == 42


def test_pickle_model(RefSimulator, seed):
    t_run = 0.5
    simseed = seed + 1

    with nengo.Network(seed=seed) as network:
        u = nengo.Node(nengo.processes.WhiteSignal(t_run, 5))
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        nengo.Connection(u, a, synapse=None)
        nengo.Connection(a, b, function=np.square)
        up = nengo.Probe(u, synapse=0.01)
        ap = nengo.Probe(a, synapse=0.01)
        bp = nengo.Probe(b, synapse=0.01)

    with RefSimulator(network, seed=simseed) as sim:
        sim.run(t_run)
        t0, u0, a0, b0 = sim.trange(), sim.data[up], sim.data[ap], sim.data[bp]
        pkls = pickle.dumps(dict(model=sim.model, up=up, ap=ap, bp=bp))

    # reload model
    del network, sim, up, ap, bp
    pkl = pickle.loads(pkls)
    up, ap, bp = pkl['up'], pkl['ap'], pkl['bp']

    with RefSimulator(None, model=pkl['model'], seed=simseed) as sim:
        sim.run(t_run)
        t1, u1, a1, b1 = sim.trange(), sim.data[up], sim.data[ap], sim.data[bp]

    tols = dict(atol=1e-5)
    assert np.allclose(t1, t0, **tols)
    assert np.allclose(u1, u0, **tols)
    assert np.allclose(a1, a0, **tols)
    assert np.allclose(b1, b0, **tols)


def test_copy_convolution():
    x = nengo.Convolution(1, (2, 3, 4), channels_last=False)
    y = copy(x)

    assert x.n_filters == y.n_filters
    assert x.input_shape == y.input_shape
    assert x.channels_last == y.channels_last
