import pytest

from nengo.builder.node import build_node
from nengo.exceptions import BuildError


def test_build_node_error():
    """Tests a build error for the build_node function"""

    class FakeOutput:
        astype = 0

    class FakeNode:
        output = FakeOutput()
        size_in = 0

    model = 0
    node = FakeNode

    with pytest.raises(BuildError):
        build_node(model, node)
