import pytest

import nengo
from nengo import spa


def test_spa_verification(Simulator, seed, plt):
    d = 16

    model = spa.SPA(seed=seed)
    # build a normal model that shouldn't raise a warning
    with model:
        model.buf = spa.Buffer(d)
        model.input_node = spa.Input(buf='B')
        # make sure errors aren't fired for non-spa modules
        prod = nengo.networks.Product(10, 2)  # noqa: F841

    with pytest.raises(ValueError):
        model = spa.SPA(seed=seed)
        # build a model that should raise a warning because no variable
        with model:
            model.buf = spa.Buffer(d)
            spa.Input(buf='B')

    with pytest.raises(ValueError):
        model = spa.SPA(seed=seed)
        # build a model that should raise a warning because no attribute
        # for the input
        with model:
            model.buf = spa.Buffer(d)
            input_node = spa.Input(buf='B')
            input_node.label = "woop"

    with pytest.raises(ValueError):
        model = spa.SPA(seed=seed)
        # build a model that should raise a warning because no attribute
        # for the buffer
        with model:
            buf = spa.Buffer(d)
            model.input_node = spa.Input(buf='B')
            buf.label = "woop"


def test_spa_module_exception():
    class MyException(Exception):
        pass

    class TestModule(spa.module.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            raise MyException()

    with pytest.raises(MyException):
        with spa.SPA() as model:
            model.test = TestModule()
