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

    with pytest.raises(KeyError):
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


def test_spa_get():
    D = 16
    model = spa.SPA()
    with model:
        model.buf1 = spa.State(D)
        model.buf2 = spa.State(D)
        model.compare = spa.Compare(D)

    assert model.get_module('buf1') is model.buf1
    assert model.get_module('buf1_default') is model.buf1
    assert model.get_module('buf2') is model.buf2
    assert model.get_module_input('buf1')[0] is model.buf1.input
    assert model.get_module_output('buf1')[0] is model.buf1.output
    assert model.get_module_input('compare_A')[0] is model.compare.inputA
    assert model.get_module_input('compare_B')[0] is model.compare.inputB

    with pytest.raises(KeyError):
        model.get_module('dummy')
    with pytest.raises(KeyError):
        model.get_module_input('dummy')
    with pytest.raises(KeyError):
        model.get_module_output('dummy')
    with pytest.raises(KeyError):
        model.get_module_input('buf1_A')
    with pytest.raises(KeyError):
        model.get_module_input('compare')
