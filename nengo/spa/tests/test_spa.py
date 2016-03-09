import pytest

import nengo
from nengo import spa
from nengo.exceptions import SpaModuleError
from nengo.utils.testing import warns


def test_spa_verification(seed, plt):
    d = 16

    model = spa.SPA(seed=seed)

    # building a normal model that shouldn't raise a warning
    with model:
        model.buf = spa.Buffer(d)
        model.input_node = spa.Input(buf='B')
        # make sure errors aren't fired for non-spa modules
        prod = nengo.networks.Product(10, 2)  # noqa: F841
        model.int_val = 1

        # reassignment is fine for non-modules
        model.int_val = 2

    # reassignment of modules should throw an error
    with pytest.raises(ValueError):
        with model:
            model.buf = spa.State(d, feedback=1)

    with pytest.raises(ValueError):
        model = spa.SPA(seed=seed)
        # build a model that should raise an error because no variable
        with model:
            model.buf = spa.Buffer(d)
            spa.Input(buf='B')

    with pytest.raises(ValueError):
        model = spa.SPA(seed=seed)
        # build a model that should raise an error because no input attribute
        with model:
            model.buf = spa.Buffer(d)
            input_node = spa.Input(buf='B')
            input_node.label = "woop"

    with pytest.raises(SpaModuleError):
        model = spa.SPA(seed=seed)
        # build a model that should raise an error because no buf attribute
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

    with pytest.raises(SpaModuleError):
        model.get_module('dummy')
    with pytest.raises(SpaModuleError):
        model.get_module_input('dummy')
    with pytest.raises(SpaModuleError):
        model.get_module_output('dummy')
    with pytest.raises(SpaModuleError):
        model.get_module_input('buf1_A')
    with pytest.raises(SpaModuleError):
        model.get_module_input('compare')


def test_spa_vocab():
    # create a model without a vocab and check that it is empty
    model = spa.SPA()
    assert model._default_vocabs == {}

    # create a model with a vocab and check that it's filled
    va = spa.Vocabulary(16)
    va.parse("PANTS")
    vb = spa.Vocabulary(32)
    vb.parse("SHOES")
    model = spa.SPA(vocabs=[va, vb])
    assert model._default_vocabs[16].keys == ["PANTS"]
    assert model._default_vocabs[32].keys == ["SHOES"]

    # warning on vocabs with duplicate dimensions
    vc = spa.Vocabulary(16)
    vc.parse("SOCKS")
    with warns(UserWarning):
        model = spa.SPA(vocabs=[va, vb, vc])
    assert model._default_vocabs[16].keys == ["SOCKS"]
    assert model._default_vocabs[32].keys == ["SHOES"]
