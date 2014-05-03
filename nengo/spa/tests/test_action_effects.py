import pytest

import nengo
from nengo.spa.action_objects import Symbol
from nengo.spa.action_effect import SourceWithAddition, Effect


A = Symbol('A')
B = Symbol('B')
x = SourceWithAddition('x')
y = SourceWithAddition('y')


def test_combined_source():
    assert str(x * y) == '((x) * (y)) * 1'
    assert str(x * 2 * y) == '((2 * x) * (y)) * 1'
    assert str((x * y) * 2) == '((x) * (y)) * 2'
    assert str((x * y) * A) == '((x) * (y)) * A'
    assert str((-x * A) * y) == '(((-1 * A) * x) * (y)) * 1'
    assert str(x * y + x) == '((x) * (y)) * 1 + x'
    assert str(x * y + y * x) == '((x) * (y)) * 1 + ((y) * (x)) * 1'


def test_source_addition():
    assert str(x + y) == 'x + y'
    assert str(x + A) == 'x + A'
    assert str(A + x) == 'x + A'
    assert str(x + 1) == 'x + 1'
    assert str(1 + x) == 'x + 1'

    with pytest.raises(TypeError):
        x + []
    with pytest.raises(TypeError):
        x + "1"

    assert str(x - y) == 'x + -1 * y'
    assert str(x - A) == 'x + -A'
    assert str(A - x) == '-1 * x + A'
    assert str(1 - x) == '-1 * x + 1'
    assert str(x - 1) == 'x + -1'

    with pytest.raises(TypeError):
        x - {}
    with pytest.raises(TypeError):
        "1" - x


def test_vector_list():
    xy = x + y

    assert str(xy * 2) == '2 * x + 2 * y'
    assert str((xy * 2) * 2) == '(2 * 2) * x + (2 * 2) * y'
    assert str((xy * 2) / 0.5) == '(2 * 2) * x + (2 * 2) * y'

    assert str(2 * xy) == '2 * x + 2 * y'
    assert str(2 * (xy * 2)) == '(2 * 2) * x + (2 * 2) * y'
    assert str(2 * (2 * xy)) == '(2 * 2) * x + (2 * 2) * y'

    assert str(xy + x) == 'x + y + x'
    assert str(xy + 0.5) == 'x + y + 0.5'
    assert str(xy + A) == 'x + y + A'
    assert str(xy + xy) == 'x + y + x + y'
    assert str(xy + 2 * xy) == 'x + y + 2 * x + 2 * y'

    with pytest.raises(TypeError):
        xy + ''

    assert str(x + xy) == 'x + y + x'
    assert str(0.5 + xy) == 'x + y + 0.5'
    assert str(A + xy) == 'x + y + A'

    with pytest.raises(TypeError):
        '' + xy

    assert str(xy - x) == 'x + y + -1 * x'
    assert str(xy - 0.5) == 'x + y + -0.5'
    assert str(xy - A) == 'x + y + -A'
    assert str(xy - xy) == 'x + y + -1 * x + -1 * y'
    assert str(xy - 2 * xy) == 'x + y + -2 * x + -2 * y'

    with pytest.raises(TypeError):
        xy - ''

    assert str(x - xy) == '-1 * x + -1 * y + x'
    assert str(0.5 - xy) == '-1 * x + -1 * y + 0.5'
    assert str(A - xy) == '-1 * x + -1 * y + A'

    with pytest.raises(TypeError):
        '' - xy

    assert str(-xy) == '-1 * x + -1 * y'
    assert str(-(-xy)) == 'x + y'


def test_effect():
    e = Effect(['a', 'b'], 'm=A')
    assert str(e) == 'm=A'
    e = Effect(['a', 'b'], 'm=A, n=B')
    assert str(e) == 'm=A, n=B' or str(e) == 'n=B, m=A'
    e = Effect(['a', 'b'], 'm=a, n=b*2*A')
    assert str(e) == 'm=a, n=(2 * A) * b' or str(e) == 'n=(2 * A) * b, m=a'

    e = Effect(['a', 'b'], 'm=0')
    assert str(e) == 'm=0'

    with pytest.raises(NameError):
        Effect(['a', 'b'], 'q=z')


def test_inverted():
    assert str(~x) == '~x'
    assert str(A*~x) == 'A * ~x'
    assert str(~x*y) == '((~x) * (y)) * 1'
    assert str(~x*~y) == '((~x) * (~y)) * 1'
    assert str(x*~y) == '((x) * (~y)) * 1'
    with pytest.raises(TypeError):
        ~(2*x)
    with pytest.raises(TypeError):
        ~(A*x)
    with pytest.raises(TypeError):
        Effect(['a', 'b'], 'a = ~2*b')
    with pytest.raises(TypeError):
        Effect(['a', 'b'], 'a = ~2*C*b')


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
