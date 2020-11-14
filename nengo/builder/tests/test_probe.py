import pytest

import nengo
from nengo.builder.probe import SimProbe
from nengo.exceptions import BuildError


def test_signal_probe(seed):
    """tests the function signal_probe"""

    # --- test probing wrong type
    with nengo.Network() as net:
        node = nengo.Node(0)
        p = nengo.Probe(node)

    with nengo.Simulator(net) as sim:
        probe_ops = [op for op in sim.model.operators if isinstance(op, SimProbe)]
        assert len(probe_ops) == 1
        assert probe_ops[0].signal is sim.model.sig[node]["out"]

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
