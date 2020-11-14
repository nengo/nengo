import pytest

from nengo.builder.processes import SimProcess
from nengo.builder.tests.test_operator import _test_operator_arg_attributes


def test_simprocess():
    argnames = ["process", "input", "output", "t"]
    non_signals = ["process"]
    _, sim = _test_operator_arg_attributes(
        SimProcess, argnames, non_signals=non_signals
    )
    assert str(sim) == "SimProcess{process, input -> output}"

    with pytest.raises(ValueError, match="Unrecognized mode"):
        _test_operator_arg_attributes(SimProcess, argnames, args={"mode": "badval"})
