import pytest

from nengo.builder.learning_rules import SimVoja, build_learning_rule
from nengo.exceptions import BuildError


def test_simvoja():

    mysim = SimVoja(1, 2, 3, 4, 5, 6, 7)
    assert mysim.weights == 3

    assert str(mysim._descstr).startswith(
        "<bound method SimVoja._descstr of <SimVoja  at "
    )


def test_build_learning_rule():
    class FakeTarget:
        shape = (1, 2)

    class FakeSig:
        def __getitem__(self, other):
            return {"encoders": FakeTarget()}

    class FakeParams:
        def __getitem__(self, other):
            return True

        def __setitem__(self, a, b):
            return True

    class FakeModel:
        def fake_add_op(self, other):
            return True

        sig = FakeSig()
        add_op = fake_add_op
        params = FakeParams()

        def fake_build(self, a, b):
            return True

        build = fake_build

    class FakeEnsembleHolder:
        ensemble = 0

    class FakeConnection:
        is_decoded = False
        post_obj = FakeEnsembleHolder()

    class FakeRule:
        def __init__(self, mod):
            self.modifies = mod

        connection = FakeConnection()
        learning_rule_type = 0

    with pytest.raises(ValueError):
        build_learning_rule(FakeModel(), FakeRule("encoders"))
    with pytest.raises(BuildError):
        build_learning_rule(FakeModel(), FakeRule("Unknown target"))
