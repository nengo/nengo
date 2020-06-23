import pytest
import nengo

from nengo.builder.probe import signal_probe, build_probe
from nengo.exceptions import BuildError
from nengo.builder import Model


def test_signal_probe(seed):
    """tests the function signal_probe"""
    func = lambda x: x ** 2

    with nengo.Network(seed=seed):
        conn = nengo.Connection(
            nengo.Ensemble(60, 1), nengo.Ensemble(50, 1), function=func
        )

    model = Model()
    with pytest.raises(BuildError):
        conn.obj = 1
        probe = build_probe(model, conn)

    class FakeObj:
        pass

    class FakeProbe:
        obj = FakeObj()
        slice = 0  # not none
        synapse = None

    probe = FakeProbe()
    key = 1

    # code is looking for an indexerror, but keyerror is given
    # if line above this is commented out
    with pytest.raises(BuildError):
        signal_probe(model, key, probe)

    model.sig[probe.obj][key] = [1, 2]
    signal_probe(model, key, probe)

    assert model.sig[probe]["in"] == 1
