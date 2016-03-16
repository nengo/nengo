import numpy as np
import pytest

import nengo
from nengo.exceptions import NetworkContextError
from nengo.params import params
from nengo.utils.compat import pickle


def assert_is_copy(cp, original):
    assert cp is not original  # ensures separate parameters
    for param in params(cp):
        param_inst = getattr(cp, param)
        if isinstance(param_inst, nengo.solvers.Solver) or isinstance(
                param_inst, nengo.base.NengoObject):
            assert param_inst is getattr(original, param)
        else:
            assert getattr(cp, param) == getattr(original, param)


def assert_is_deepcopy(cp, original):
    assert cp is not original  # ensures separate parameters
    for param in params(cp):
        param_inst = getattr(cp, param)
        if isinstance(param_inst, nengo.solvers.Solver) or isinstance(
                param_inst, nengo.base.NengoObject):
            assert_is_copy(param_inst, getattr(original, param))
        else:
            assert param_inst == getattr(original, param)


class CopyTest(object):
    def create_original(self):
        raise NotImplementedError()

    def assert_is_copy(self, cp, original):
        assert_is_copy(cp, original)

    def test_copy_in_network(self):
        original = self.create_original()

        with nengo.Network() as model:
            cp = original.copy()
        assert cp in model.all_objects

        self.assert_is_copy(cp, original)

    def test_copy_in_network_without_adding(self):
        original = self.create_original()

        with nengo.Network() as model:
            cp = original.copy(add_to_container=False)
        assert cp not in model.all_objects

        self.assert_is_copy(cp, original)

    def test_copy_outside_network(self):
        original = self.create_original()
        with pytest.raises(NetworkContextError):
            original.copy()

    def test_copy_outside_network_without_adding(self):
        original = self.create_original()
        cp = original.copy(add_to_container=False)
        self.assert_is_copy(cp, original)


class PickleTest(object):
    def create_original(self):
        raise NotImplementedError()

    def assert_is_unpickled(self, cp, original):
        assert_is_deepcopy(cp, original)

    def test_pickle_roundtrip(self):
        original = self.create_original()
        cp = pickle.loads(pickle.dumps(original))
        self.assert_is_unpickled(cp, original)


class TestCopyEnsemble(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network():
            e = nengo.Ensemble(10, 1, radius=2.)
        return e

    def test_neurons_reference_copy(self):
        original = self.create_original()
        cp = original.copy(add_to_container=False)
        assert original.neurons.ensemble is original
        assert cp.neurons.ensemble is cp


class TestCopyProbe(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network():
            e = nengo.Ensemble(10, 1)
            p = nengo.Probe(e, synapse=0.01)
        return p


class TestCopyNode(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network():
            n = nengo.Node(np.min, size_in=2, size_out=2)
        return n


class TestCopyConnection(CopyTest, PickleTest):
    def create_original(self):
        with nengo.Network(self):
            e1 = nengo.Ensemble(10, 1)
            e2 = nengo.Ensemble(10, 1)
            c = nengo.Connection(e1, e2, transform=2.)
        return c

# Network

# Process
# Learning rule
# Synapse


# copy, copy with add to network, pickle and unpickle
