import pytest
from nengo.builder.processes import SimProcess


def test_simprocess(seed, rng):
    """tests the SimProcess class"""
    process = 0
    input = 0
    output = 0
    t = 0
    with pytest.raises(ValueError):
        SimProcess(process, input, output, t, mode="not a mode")

    simp = SimProcess(process, input, output, t)

    simp.sets = []

    assert simp.output is None
