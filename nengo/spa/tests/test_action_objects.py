import pytest

import nengo
from nengo.spa.action_objects import Symbol, Source


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

    with pytest.raises(TypeError):
        s + s_q
    with pytest.raises(TypeError):
        s * "A"


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
