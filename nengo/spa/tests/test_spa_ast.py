import pytest

from nengo import spa
from nengo.exceptions import SpaTypeError
from nengo.spa.actions import Parser
from nengo.spa.spa_ast import (
    ApproxInverse, DotProduct, Negative, Module, Product, TScalar, TVocabulary,
    Sum, Symbol)


def test_scalar():
    ast = Parser().parse('1')
    assert ast == 1


def test_symbol():
    ast = Parser().parse('A')
    assert ast == Symbol('A')
    assert str(ast) == 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    ast = Parser().parse('A')
    with pytest.raises(SpaTypeError):
        ast.infer_types(None, TScalar)


def test_module():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse('state')
    assert ast == Module('state')
    assert str(ast) == 'state'
    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]

    with spa.Module() as model:
        with spa.Module() as model.module:
            model.module.state = spa.State(d)

    ast = Parser().parse('module.state')
    assert ast == Module('module.state')
    assert str(ast) == 'module.state'
    ast.infer_types(model, None)
    assert ast.type.vocab == model.module.state.vocabs[d]


def test_scalar_multiplication():
    ast = Parser().parse('2 * A')
    assert ast == Product(2, Symbol('A'))
    assert str(ast) == '(2 * A)'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    ast = Parser().parse('A * 2')
    assert ast == Product(Symbol('A'), 2)
    assert str(ast) == '(A * 2)'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse('2 * state')
    assert ast == Product(2, Module('state'))
    assert str(ast) == '(2 * state)'

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]

    ast = Parser().parse('state * 2')
    assert ast == Product(Module('state'), 2)
    assert str(ast) == '(state * 2)'

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [('+', Sum), ('*', Product)])
def test_binary_operations(symbol, klass):
    ast = Parser().parse('A {} B'.format(symbol))
    assert ast == klass(Symbol('A'), Symbol('B'))
    assert str(ast) == '(A {} B)'.format(symbol)

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)
        model.state2 = spa.State(d)

    ast = Parser().parse('state {} B'.format(symbol))
    assert ast == klass(Module('state'), Symbol('B'))
    assert str(ast) == '(state {} B)'.format(symbol)

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type.vocab == model.state.vocabs[d]

    ast = Parser().parse('A {} state'.format(symbol))
    assert ast == klass(Symbol('A'), Module('state'))
    assert str(ast) == '(A {} state)'.format(symbol)

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type.vocab == model.state.vocabs[d]

    ast = Parser().parse('state {} state2'.format(symbol))
    assert ast == klass(Module('state'), Module('state2'))
    assert str(ast) == '(state {} state2)'.format(symbol)

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [
    ('~', ApproxInverse), ('-', Negative)])
def test_unary(symbol, klass):
    ast = Parser().parse(symbol + 'A')
    assert ast == klass(Symbol('A'))
    assert str(ast) == symbol + 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse(symbol + 'state')
    assert ast == klass(Module('state'))
    assert str(ast) == symbol + 'state'

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]


def test_complex_epressions():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse('~(A - B * state)')
    assert ast == ApproxInverse(Sum(
        Symbol('A'), Negative(Product(Symbol('B'), Module('state')))))
    assert str(ast) == '~(A + -(B * state))'

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type.vocab == model.state.vocabs[d]


def test_dot_product():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse('dot(A, state)')
    assert ast == DotProduct(Symbol('A'), Module('state'))
    assert str(ast) == 'dot(A, state)'

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type == TScalar

    ast = Parser().parse('2 * dot(A, state) + 1')
    assert ast == Sum(Product(2, DotProduct(Symbol('A'), Module('state'))), 1)
    assert str(ast) == '((2 * dot(A, state)) + 1)'

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type == TScalar
