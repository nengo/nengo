from nengo.objects import Direct
from nengo import Model

"""
SimModel is made up of a bunch of sets.
This test suite makes sure that the simulator objects
interact with these sets as we expect them to.
"""

def test_set():
    m = Model("test_set")
    f = lambda x: x
    nl1 = Direct(1, 1, f)
    nl2 = Direct(2, 2, f)
    m.add(nl1)
    assert len(m.nonlinearities) == 1
    m.add(nl1)
    assert len(m.nonlinearities) ==  1  # Already in there
    m.add(nl2)
    assert len(m.nonlinearities) == 2
    m.add(Direct(1, 1, f))  # Same params as nl1, but different
    assert len(m.nonlinearities) ==  3
