import numpy as np
import pytest

import nengo


def test_seeding():
    """Test that setting the model seed fixes everything"""

    ### TODO: this really just checks random parameters in ensembles.
    ###   Are there other objects with random parameters that should be
    ###   tested? (Perhaps initial weights of learned connections)

    m = nengo.Model('test_seeding')
    input = nengo.Node(output=1, label='input')
    A = nengo.Ensemble(nengo.LIF(40), 1, label='A')
    B = nengo.Ensemble(nengo.LIF(20), 1, label='B')
    nengo.Connection(input, A)
    C = nengo.Connection(A, B, function=lambda x: x ** 2)

    m.seed = 872
    m1 = nengo.Simulator(m).model.params
    m2 = nengo.Simulator(m).model.params
    m.seed = 873
    m3 = nengo.Simulator(m).model.params

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = (np.all(getattr(obj1, attr) == getattr(obj2, attr))
                     if equal else
                     np.any(getattr(obj1, attr) != getattr(obj2, attr)))
            if not check:
                print(attr, getattr(obj1, attr))
                print(attr, getattr(obj2, attr))
            assert check

    ens_attrs = ('encoders', 'eval_points', 'max_rates',
                 'intercepts', 'scaled_encoders')
    As = [mi[A] for mi in [m1, m2, m3]]
    Bs = [mi[B] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], ens_attrs)
    compare_objs(Bs[0], Bs[1], ens_attrs)
    compare_objs(As[0], As[2], ens_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], ens_attrs, equal=False)

    neur_attrs = ('gain', 'bias')
    As = [mi[A.neurons] for mi in [m1, m2, m3]]
    Bs = [mi[B.neurons] for mi in [m1, m2, m3]]
    compare_objs(As[0], As[1], neur_attrs)
    compare_objs(Bs[0], Bs[1], neur_attrs)
    compare_objs(As[0], As[2], neur_attrs, equal=False)
    compare_objs(Bs[0], Bs[2], neur_attrs, equal=False)

    conn_attrs = ('decoders', 'eval_points')
    Cs = [mi[C] for mi in [m1, m2, m3]]
    compare_objs(Cs[0], Cs[1], conn_attrs)
    compare_objs(Cs[0], Cs[2], conn_attrs, equal=False)

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
