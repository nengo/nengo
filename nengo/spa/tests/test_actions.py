import pytest

from nengo import spa
from nengo.spa.actions import Expression, Effect, Action, Actions


def test_expression():
    c = Expression(['a', 'b'], 'dot(a, A)')
    assert str(c.expression) == 'dot(a, A)'

    c = Expression(['a', 'b'], '0.5*(2*dot(a, A)-dot(b,B))-2')
    assert str(c.expression) == 'dot(a, A) + -0.5 * dot(b, B) + -2'

    c = Expression(['a'], '1')
    assert str(c.expression) == '1'

    with pytest.raises(NameError):
        c = Expression(['a', 'b'], 'dot(c, C)')


def test_scalars():
    c = Expression([], '1')
    assert str(c.expression) == '1'

    c = Expression([], '1 - 1')
    assert str(c.expression) == '0'

    c = Expression(['x'], 'dot(1, x)')
    assert str(c.expression) == 'dot(1, x)'

    c = Expression(['x'], 'dot(x, -1) + 1')
    assert str(c.expression) == 'dot(x, -1) + 1'

    c = Expression(['a', 'b'], '2*dot(a, 1) - dot(b, -1) + dot(a, b)')
    assert str(c.expression) == '2 * dot(a, 1) + -dot(b, -1) + dot(a, b)'

    c = Expression(['a', 'b'], 'a*b - 1 + 2*b')
    assert str(c.expression) == '((a) * (b)) * 1 + -1 + 2 * b'


def test_effect():
    e = Effect(['a', 'b'], ['m'], 'm=A')
    assert str(e) == 'm=A'
    e = Effect(['a', 'b'], ['m', 'n'], 'm=A, n=B')
    assert str(e) == 'm=A, n=B'
    e = Effect(['a', 'b'], ['m', 'n'], 'm=a, n=b*2*A')
    assert str(e) == 'm=a, n=(2 * A) * b'
    e = Effect(['a', 'b'], ['m'], 'm=0')
    assert str(e) == 'm=0'

    # Check that multiple lvalue=rvalue parsing is working with commas
    e = Effect(['a', 'b'], ['x', 'y', 'z'], 'x=a,y=dot(a,b),z=b')
    assert str(e) == 'x=a, y=dot(a, b), z=b'
    e = Effect(['a', 'b'], ['foo', 'bar'], '  foo = dot(a, b)  , bar = b')
    assert str(e) == 'foo=dot(a, b), bar=b'

    with pytest.raises(NameError):
        Effect(['a', 'b'], ['q'], 'q=z')

    with pytest.raises(ValueError):
        Effect(['a', 'b'], ['q'], 'q=a, q=b')  # lvalue appears twice


def test_inverted():
    with pytest.raises(ValueError):
        Effect(['b'], ['a'], 'a = ~2*b')
    with pytest.raises(ValueError):
        Effect(['b'], ['a'], 'a = ~2*C*b')


def test_action():
    a = Action(['vision', 'memory'], ['motor', 'memory'],
               'dot(vision, DOG) --> motor=vision', 'test_rule')
    assert str(a.condition) == 'dot(vision, DOG)'
    assert str(a.effect) == 'motor=vision'

    a = Action(['vision', 'memory'], ['motor', 'memory'],
               'motor=vision*A', 'test_rule')
    assert a.condition is None
    assert str(a.effect) == 'motor=A * vision'

    with pytest.raises(NameError):
        a = Action(['vision', 'memory'], ['motor', 'memory'],
                   'motor=vis*A', 'test_action')

    with pytest.raises(NameError):
        a = Action(['vision', 'memory'], ['motor', 'memory'],
                   '0.5 --> motor=vis*A', 'test_action')

    with pytest.raises(NameError):
        a = Action(['vision', 'memory'], ['motor', 'memory'],
                   '0.5*dot(mem, a) --> motor=B', name=None)

    with pytest.raises(TypeError):
        a = Action(['vision', 'memory'], ['motor', 'memory'],
                   '0.5*dot(memory+1, vision) --> motor=B', name='test_action')

    with pytest.raises(NameError):
        a = Action(['vision', 'memory'], ['motor', 'memory'],
                   'motor2=B', name='test_action')
    with pytest.raises(NameError):
        a = Action(['vision', 'memory'], ['motor', 'memory'],
                   'motor=A, motor2=B', name='test_action')


def test_actions():
    a = Actions(
        'dot(state, A) --> state=B',
        'dot(state, B) --> state=A',
        default='1.0 --> state=C',
    )
    assert a.count == 3

    model = spa.SPA()
    with model:
        model.state = spa.Buffer(16)
    a.process(model)
    assert str(a.actions[0].condition) == 'dot(state, A)'
    assert str(a.actions[0].effect) == 'state=B'
    assert str(a.actions[1].condition) == 'dot(state, B)'
    assert str(a.actions[1].effect) == 'state=A'
    assert str(a.actions[2].condition) == '1.0'
    assert str(a.actions[2].effect) == 'state=C'
