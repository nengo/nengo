import pytest

import nengo
from nengo.spa.action_objects import Symbol, Source, DotProduct

A = Symbol('A')
B = Symbol('B')
x = Source('x')
y = Source('y')


def test_symbol():
    A = Symbol('A')
    assert A.symbol == 'A'
    B = Symbol('B')
    assert str(B) == 'B'

    assert str(A + B) == '(A + B)'
    with pytest.raises(TypeError):
        A + "A"

    assert str(A - B) == '(A - B)'
    with pytest.raises(TypeError):
        A - []

    assert str(A * B) == '(A * B)'
    assert str(A * 1) == 'A'
    assert str(1 * A) == 'A'
    assert str(1.0 * A) == 'A'
    assert str(A * 1.0) == 'A'
    assert str(A * 1.5) == '(A * 1.5)'
    assert str(A * 2) == '(A * 2)'
    assert str(A * Symbol('1')) == 'A'
    assert str(Symbol('1') * A) == 'A'

    with pytest.raises(TypeError):
        A * {}

    assert str(~A) == '~A'
    assert str(~(~A)) == 'A'

    assert str(-A) == '-A'
    assert str(-(-A)) == 'A'

    assert str(A*B+(B*~A)*0.5-A) == '(((A * B) + ((B * ~A) * 0.5)) - A)'


def test_source():
    A = Symbol('A')
    s = Source('s')
    assert s.name == 's'
    assert str(s.transform) == '1'
    assert str(s) == 's'
    s_q = Source('s', Symbol('Q'))
    assert s_q.name == 's'
    assert str(s_q.transform) == 'Q'
    assert str(s_q) == 'Q * s'

    assert str(s * 1.5) == '1.5 * s'
    assert str(s * A) == 'A * s'
    assert str(s_q * A) == '(Q * A) * s'

    assert str(1.5 * s) == '1.5 * s'
    assert str(A * s) == 'A * s'
    assert str(A * s_q) == '(Q * A) * s'

    assert str(-s) == '-1 * s'
    assert str(-(-s)) == 's'
    assert str(-s_q) == '-Q * s'

    assert str(s + s_q) == 's + Q * s'

    with pytest.raises(TypeError):
        s * "A"


def test_inverted():
    assert str(~x) == '~x'
    assert str(A*~x) == 'A * ~x'
    assert str(~x*y) == '((~x) * (y)) * 1'
    assert str(~x*~y) == '((~x) * (~y)) * 1'
    assert str(x*~y) == '((x) * (~y)) * 1'
    with pytest.raises(ValueError):
        ~(2*x)
    with pytest.raises(ValueError):
        ~(A*x)


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
    assert str(DotProduct(x, y) + A) == 'dot(x, y) + A'
    assert str(B + DotProduct(x, y)) == 'dot(x, y) + B'

    assert str(-DotProduct(x, y)) == '-dot(x, y)'
    assert str(-(-DotProduct(x, y))) == 'dot(x, y)'

    assert str(DotProduct(x, y) - 1) == 'dot(x, y) + -1'
    assert str(1 - DotProduct(x, y)) == '-dot(x, y) + 1'
    assert str(DotProduct(x, y) - DotProduct(B, y)) == 'dot(x, y) + -dot(B, y)'
    assert str(DotProduct(x, y) - A) == 'dot(x, y) + -A'
    assert str(B - DotProduct(x, y)) == '-dot(x, y) + B'


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
    assert str(xy + (Ay + xy)) == 'dot(x, y) + dot(A, y) + dot(x, y)'

    assert str(xy + (Ay - xy)) == 'dot(x, y) + dot(A, y) + -dot(x, y)'
    assert str((xy + Ay) - xy) == 'dot(x, y) + dot(A, y) + -dot(x, y)'
    assert str(xy - (Ay + xy)) == 'dot(x, y) + -dot(A, y) + -dot(x, y)'

    assert str((xy + Ay) - 1) == 'dot(x, y) + dot(A, y) + -1'
    assert str((xy + Ay) + 1) == 'dot(x, y) + dot(A, y) + 1'
    assert str(1 - (xy + Ay)) == '-dot(x, y) + -dot(A, y) + 1'
    assert str(1 + (xy + Ay)) == 'dot(x, y) + dot(A, y) + 1'

    assert str((xy + Ay) + A) == 'dot(x, y) + dot(A, y) + A'


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

    assert str(x + xy) == 'x + x + y'
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

    assert str(x - xy) == 'x + -1 * x + -1 * y'
    assert str(0.5 - xy) == '-1 * x + -1 * y + 0.5'
    assert str(A - xy) == '-1 * x + -1 * y + A'

    with pytest.raises(TypeError):
        '' - xy

    assert str(-xy) == '-1 * x + -1 * y'
    assert str(-(-xy)) == 'x + y'


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
