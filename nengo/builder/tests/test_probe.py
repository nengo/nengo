import pytest
import nengo

from nengo.builder.probe import signal_probe, build_probe
from nengo.exceptions import BuildError
from nengo.builder import Model


def test_signal_probe(seed):
    """tests the function signal_probe"""

    # --- test probing wrong type
    with nengo.Network() as net:
        p = nengo.Probe(nengo.Node(0))

    # hack to change probe target without tripping API validation
    nengo.Probe.target.data[p] = 1

    with pytest.raises(BuildError, match="Type .* is not probeable"):
        with nengo.Simulator(net):
            pass

    # --- test probing wrong attribute
    with nengo.Network() as net:
        p = nengo.Probe(nengo.Node(0))

    # hack to change probe attr without tripping API validation
    nengo.Probe.attr.data[p] = "batattr"

    with pytest.raises(BuildError, match="Attribute .* is not probeable"):
        with nengo.Simulator(net):
            pass
