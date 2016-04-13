import pytest

from nengo import spa
from nengo.exceptions import SpaTypeError
from nengo.spa.actions import Parser
from nengo.spa.spa_ast import (
    Action, ApproxInverse, DotProduct, Effect, Negative, Module,
    Product, Sink, Sum, Symbol, TAction, TEffect, TScalar, TVocabulary)


def test_scalar():
    ast = Parser().parse_expr('1')
    assert ast == 1


def test_symbol():
    ast = Parser().parse_expr('A')
    assert ast == Symbol('A')
    assert str(ast) == 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    ast = Parser().parse_expr('A')
    with pytest.raises(SpaTypeError):
        ast.infer_types(None, TScalar)


def test_module():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_expr('state')
    assert ast == Module('state')
    assert str(ast) == 'state'
    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]

    with spa.Module() as model:
        with spa.Module() as model.module:
            model.module.state = spa.State(d)

    ast = Parser().parse_expr('module.state')
    assert ast == Module('module.state')
    assert str(ast) == 'module.state'
    ast.infer_types(model, None)
    assert ast.type.vocab == model.module.state.vocabs[d]


def test_scalar_multiplication():
    ast = Parser().parse_expr('2 * A')
    assert ast == Product(2, Symbol('A'))
    assert str(ast) == '(2 * A)'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    ast = Parser().parse_expr('A * 2')
    assert ast == Product(Symbol('A'), 2)
    assert str(ast) == '(A * 2)'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_expr('2 * state')
    assert ast == Product(2, Module('state'))
    assert str(ast) == '(2 * state)'

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]

    ast = Parser().parse_expr('state * 2')
    assert ast == Product(Module('state'), 2)
    assert str(ast) == '(state * 2)'

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [('+', Sum), ('*', Product)])
def test_binary_operations(symbol, klass):
    ast = Parser().parse_expr('A {} B'.format(symbol))
    assert ast == klass(Symbol('A'), Symbol('B'))
    assert str(ast) == '(A {} B)'.format(symbol)

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)
        model.state2 = spa.State(d)

    ast = Parser().parse_expr('state {} B'.format(symbol))
    assert ast == klass(Module('state'), Symbol('B'))
    assert str(ast) == '(state {} B)'.format(symbol)

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type.vocab == model.state.vocabs[d]

    ast = Parser().parse_expr('A {} state'.format(symbol))
    assert ast == klass(Symbol('A'), Module('state'))
    assert str(ast) == '(A {} state)'.format(symbol)

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type.vocab == model.state.vocabs[d]

    ast = Parser().parse_expr('state {} state2'.format(symbol))
    assert ast == klass(Module('state'), Module('state2'))
    assert str(ast) == '(state {} state2)'.format(symbol)

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]


@pytest.mark.parametrize('symbol, klass', [
    ('~', ApproxInverse), ('-', Negative)])
def test_unary(symbol, klass):
    ast = Parser().parse_expr(symbol + 'A')
    assert ast == klass(Symbol('A'))
    assert str(ast) == symbol + 'A'

    vocab_type = TVocabulary(spa.Vocabulary(16))
    ast.infer_types(None, vocab_type)
    assert ast.type == vocab_type

    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_expr(symbol + 'state')
    assert ast == klass(Module('state'))
    assert str(ast) == symbol + 'state'

    ast.infer_types(model, None)
    assert ast.type.vocab == model.state.vocabs[d]


def test_complex_epressions():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_expr('~(A - B * state)')
    assert ast == ApproxInverse(Sum(
        Symbol('A'), Negative(Product(Symbol('B'), Module('state')))))
    assert str(ast) == '~(A + -(B * state))'

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type.vocab == model.state.vocabs[d]


def test_dot_product():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_expr('dot(A, state)')
    assert ast == DotProduct(Symbol('A'), Module('state'))
    assert str(ast) == 'dot(A, state)'

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type == TScalar

    ast = Parser().parse_expr('2 * dot(A, state) + 1')
    assert ast == Sum(Product(2, DotProduct(Symbol('A'), Module('state'))), 1)
    assert str(ast) == '((2 * dot(A, state)) + 1)'

    ast.infer_types(model, TVocabulary(model.state.vocabs[d]))
    assert ast.type == TScalar


def test_effect():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_effect('state = A')
    assert ast == Effect(Sink('state'), Symbol('A'))
    assert str(ast) == 'state = A'
    assert ast.type == TEffect


def test_action():
    d = 16
    with spa.Module() as model:
        model.state = spa.State(d)

    ast = Parser().parse_action('dot(state, A) --> state = B')
    assert ast == Action(
        DotProduct(Module('state'), Symbol('A')),
        Effect(Sink('state'), Symbol('B')))
    assert str(ast) == 'dot(state, A) --> state = B'
    assert ast.type == TAction


# def test_actions():
#     a = Actions(
#         'dot(state, A) --> state=B',
#         'dot(state, B) --> state=A',
#         default='1.0 --> state=C',
#     )
#     assert a.count == 3
#
#     model = spa.Module()
#     with model:
#         model.state = spa.State(16)
#     a.process(model)
#     assert str(a.actions[0].condition) == 'dot(state, A)'
#     assert str(a.actions[0].effect) == 'state=B'
#     assert str(a.actions[1].condition) == 'dot(state, B)'
#     assert str(a.actions[1].effect) == 'state=A'
#     assert str(a.actions[2].condition) == '1.0'
#     assert str(a.actions[2].effect) == 'state=C'
#
#     a.add('dot(state, D) --> state=E')
#     a.add(added='dot(state, E) --> state=F')
#     a.process(model)
#     assert str(a.actions[2].condition) == 'dot(state, D)'
#     assert str(a.actions[2].effect) == 'state=E'
#     assert str(a.actions[3].condition) == 'dot(state, E)'
#     assert str(a.actions[3].effect) == 'state=F'
