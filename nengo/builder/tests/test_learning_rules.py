import pytest

from nengo.builder.learning_rules import SimBCM, SimOja, SimPES, SimVoja
from nengo.builder.tests.test_operator import _test_operator_arg_attributes
from nengo.exceptions import BuildError


def test_simpes():
    argnames = [
        "pre_filtered",
        "error",
        "delta",
        "learning_rate",
        "encoders",
    ]
    args, sim = _test_operator_arg_attributes(SimPES, argnames)

    assert str(sim) == "SimPES{pre=pre_filtered, error=error -> delta}"


def test_simbcm():
    argnames = [
        "pre_filtered",
        "post_filtered",
        "theta",
        "delta",
        "learning_rate",
    ]
    args, sim = _test_operator_arg_attributes(SimBCM, argnames)

    assert str(sim) == "SimBCM{pre=pre_filtered, post=post_filtered -> delta}"


def test_simoja():
    argnames = [
        "pre_filtered",
        "post_filtered",
        "weights",
        "delta",
        "learning_rate",
        "beta",
    ]
    args, sim = _test_operator_arg_attributes(SimOja, argnames)

    assert str(sim) == "SimOja{pre=pre_filtered, post=post_filtered -> delta}"


def test_simvoja():
    argnames = [
        "pre_decoded",
        "post_filtered",
        "scaled_encoders",
        "delta",
        "scale",
        "learning_signal",
        "learning_rate",
    ]
    args, sim = _test_operator_arg_attributes(SimVoja, argnames)
    assert sim.weights is args["scaled_encoders"]

    assert str(sim) == "SimVoja{pre=pre_decoded, post=post_filtered -> delta}"
