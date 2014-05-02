import pytest

import nengo
from nengo import spa
from nengo.spa.actions import Action, Actions


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

    class Test(spa.SPA):
        def __init__(self):
            self.state = spa.Buffer(16)

    model = Test()
    a.process(model)
    assert str(a.actions[0].condition) == 'dot(state, A)'
    assert str(a.actions[0].effect) == 'state=B'
    assert str(a.actions[1].condition) == 'dot(state, B)'
    assert str(a.actions[1].effect) == 'state=A'
    assert str(a.actions[2].condition) == '1.0'
    assert str(a.actions[2].effect) == 'state=C'


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
