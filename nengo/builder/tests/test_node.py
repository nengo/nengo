import pytest

import nengo
from nengo.exceptions import BuildError


def test_build_node_error(Simulator):
    """Tests a build error for the build_node function"""

    class BadOutputType:
        pass

    with nengo.Network() as net:
        n = nengo.Node(0)

    # hack to change node type without tripping API validation
    nengo.Node.output.data[n] = BadOutputType()

    with pytest.raises(BuildError, match="Invalid node output type"):
        with nengo.Simulator(net):
            pass
