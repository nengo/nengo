import numpy as np
import pytest

from nengo.builder.processes import SimProcess
from nengo.builder.tests.test_operator import _test_operator_arg_attributes
from nengo.processes import Process


def test_simprocess():
    argnames = ["process", "input", "output", "t"]
    non_signals = ["process"]
    _, sim = _test_operator_arg_attributes(
        SimProcess, argnames, non_signals=non_signals
    )
    assert str(sim) == "SimProcess{process, input -> output}"

    with pytest.raises(ValueError, match="Unrecognized mode"):
        _test_operator_arg_attributes(SimProcess, argnames, args={"mode": "badval"})


@pytest.mark.parametrize("mode", ["set", "inc"])
@pytest.mark.parametrize("has_input", [False, True])
def test_simprocess_make_step(mode, has_input, rng):
    t0 = rng.uniform(size=1)
    in0 = rng.uniform(size=1)
    out0 = rng.uniform(size=1)
    ref = t0 + (in0 if has_input else 0) + (out0 if mode == "inc" else 0)

    signals = {"in": in0.copy(), "out": out0.copy(), "t": t0.copy()}
    sim = SimProcess(
        TimeAddProcess(),
        input="in" if has_input else None,
        output="out",
        t="t",
        mode=mode,
    )
    step = sim.make_step(signals, dt=1, rng=rng)
    step()

    assert np.array_equal(signals["out"], ref)


class TimeAddProcess(Process):
    def make_step(self, shape_in, shape_out, dt, rng, state):
        return (lambda t: t) if np.prod(shape_in) == 0 else (lambda t, x: x + t)
