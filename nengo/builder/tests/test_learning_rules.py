from nengo.builder.learning_rules import SimBCM, SimOja, SimPES, SimVoja
from nengo.builder.tests.test_operator import _test_operator_arg_attributes


def test_simpes():
    argnames = ["pre_filtered", "error", "delta", "learning_rate"]
    non_signals = ["learning_rate"]
    _, sim = _test_operator_arg_attributes(SimPES, argnames, non_signals=non_signals)

    assert str(sim) == "SimPES{pre=pre_filtered, error=error -> delta}"


def test_simbcm():
    argnames = ["pre_filtered", "post_filtered", "theta", "delta", "learning_rate"]
    non_signals = ["learning_rate"]
    _, sim = _test_operator_arg_attributes(SimBCM, argnames, non_signals=non_signals)

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
    non_signals = ["learning_rate", "beta"]
    _, sim = _test_operator_arg_attributes(SimOja, argnames, non_signals=non_signals)

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
    non_signals = ["learning_rate", "scale"]
    args, sim = _test_operator_arg_attributes(
        SimVoja, argnames, non_signals=non_signals
    )
    assert sim.weights is args["scaled_encoders"]

    assert str(sim) == "SimVoja{pre=pre_decoded, post=post_filtered -> delta}"
