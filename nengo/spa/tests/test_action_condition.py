import pytest

import nengo
from nengo.spa.action_objects import Symbol, Source
from nengo.spa.action_condition import DotProduct, Condition


A = Symbol('A')
B = Symbol('B')
x = Source('x')
y = Source('y')


def test_dotproduct():
    assert str(DotProduct(x, y)) == 'dot(x, y)'
    assert str(DotProduct(x, A)) == 'dot(x, A)'
    assert str(DotProduct(B, y)) == 'dot(B, y)'

    with pytest.raises(TypeError):
        DotProduct(B, A)
    with pytest.raises(TypeError):
        DotProduct(A, "B")
    with pytest.raises(TypeError):
        DotProduct(1, A)

    assert str(0.5*DotProduct(x, y)) == '0.5 * dot(x, y)'
    assert str(0.5 * (0.5*DotProduct(x, y))) == '0.25 * dot(x, y)'
    assert str(DotProduct(B, y)*2) == '2 * dot(B, y)'

    with pytest.raises(TypeError):
        A * DotProduct(x, y)
    with pytest.raises(TypeError):
        DotProduct(x, y) * B

    assert str(DotProduct(x, y)/2) == '0.5 * dot(x, y)'
    assert str((DotProduct(x, y)/2)/2) == '0.25 * dot(x, y)'
    assert str(2*DotProduct(B, y)/2) == 'dot(B, y)'

    with pytest.raises(TypeError):
        DotProduct(x, y) / B

    assert str(DotProduct(x, y) + 1) == 'dot(x, y) + 1'
    assert str(1 + DotProduct(x, y)) == 'dot(x, y) + 1'
    assert str(DotProduct(x, y) + DotProduct(B, y)) == 'dot(x, y) + dot(B, y)'

    with pytest.raises(TypeError):
        DotProduct(x, y) + A
    with pytest.raises(TypeError):
        B + DotProduct(x, y)

    assert str(-DotProduct(x, y)) == '-1 * dot(x, y)'
    assert str(-(-DotProduct(x, y))) == 'dot(x, y)'

    assert str(DotProduct(x, y) - 1) == 'dot(x, y) + -1'
    assert str(1 - DotProduct(x, y)) == '-1 * dot(x, y) + 1'
    assert str(DotProduct(x, y) - DotProduct(B, y)) == \
        'dot(x, y) + -1 * dot(B, y)'

    with pytest.raises(TypeError):
        DotProduct(x, y) - A
    with pytest.raises(TypeError):
        B - DotProduct(x, y)


def test_condition_list():
    xy = DotProduct(x, y)
    Ay = DotProduct(A, y)

    assert str(xy + Ay) == 'dot(x, y) + dot(A, y)'

    assert str(2*(xy + Ay)) == '2 * dot(x, y) + 2 * dot(A, y)'
    assert str((xy + Ay)*2) == '2 * dot(x, y) + 2 * dot(A, y)'
    assert str(2*(xy + Ay)*2) == '4 * dot(x, y) + 4 * dot(A, y)'

    assert str(2*(xy + Ay)/2) == 'dot(x, y) + dot(A, y)'
    assert str(2*(xy + Ay)/0.5) == '4 * dot(x, y) + 4 * dot(A, y)'

    assert str(xy + Ay + xy) == 'dot(x, y) + dot(A, y) + dot(x, y)'
    assert str((xy + Ay) + xy) == 'dot(x, y) + dot(A, y) + dot(x, y)'
    assert str(2*(xy + Ay) + xy) == '2 * dot(x, y) + 2 * dot(A, y) + dot(x, y)'
    assert str(xy + (Ay + xy)) == 'dot(A, y) + dot(x, y) + dot(x, y)'

    assert str(xy + Ay - xy) == 'dot(x, y) + dot(A, y) + -1 * dot(x, y)'
    assert str((xy + Ay) - xy) == 'dot(x, y) + dot(A, y) + -1 * dot(x, y)'
    assert str(xy - (Ay + xy)) == '-1 * dot(A, y) + -1 * dot(x, y) + dot(x, y)'

    assert str((xy + Ay) - 1) == 'dot(x, y) + dot(A, y) + -1'
    assert str((xy + Ay) + 1) == 'dot(x, y) + dot(A, y) + 1'
    assert str(1 - (xy + Ay)) == '-1 * dot(x, y) + -1 * dot(A, y) + 1'
    assert str(1 + (xy + Ay)) == 'dot(x, y) + dot(A, y) + 1'
    with pytest.raises(TypeError):
        (xy + Ay) + A


def test_condition():
    c = Condition(['a', 'b'], 'dot(a, A)')
    assert str(c.condition) == 'dot(a, A)'

    c = Condition(['a', 'b'], '0.5*(2*dot(a, A)-dot(b,B))-2')
    assert str(c.condition) == 'dot(a, A) + -0.5 * dot(b, B) + -2'

    c = Condition(['a'], '1')
    assert str(c.condition) == '1'

    with pytest.raises(NameError):
        c = Condition(['a', 'b'], 'dot(c, C)')

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
