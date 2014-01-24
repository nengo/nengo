import numpy as np
import pytest

import nengo


@pytest.mark.skipif(True, reason="The model doesn't change anymore...")
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
    nengo.Connection(A, B, function=lambda x: x ** 2)
    # input_p = nengo.Probe(input, 'output')
    # A_p = nengo.Probe(A, 'decoded_output', filter=0.01)
    # B_p = nengo.Probe(B, 'decoded_output', filter=0.01)

    m.seed = 872
    m1 = nengo.Simulator(m).model
    m2 = nengo.Simulator(m).model
    m.seed = 873
    m3 = nengo.Simulator(m).model

    def compare_objs(obj1, obj2, attrs, equal=True):
        for attr in attrs:
            check = (np.all(getattr(obj1, attr) == getattr(obj2, attr))
                     if equal else
                     np.any(getattr(obj1, attr) != getattr(obj2, attr)))
            if not check:
                print(getattr(obj1, attr))
                print(getattr(obj2, attr))
            assert check

    ens_attrs = ('encoders', 'max_rates', 'intercepts')
    A = [[o for o in mi.objs if o.label == 'A'][0] for mi in [m1, m2, m3]]
    B = [[o for o in mi.objs if o.label == 'B'][0] for mi in [m1, m2, m3]]
    compare_objs(A[0], A[1], ens_attrs)
    compare_objs(B[0], B[1], ens_attrs)
    compare_objs(A[0], A[2], ens_attrs, equal=False)
    compare_objs(B[0], B[2], ens_attrs, equal=False)

    neur_attrs = ('gain', 'bias')
    compare_objs(A[0].neurons, A[1].neurons, neur_attrs)
    compare_objs(B[0].neurons, B[1].neurons, neur_attrs)
    compare_objs(A[0].neurons, A[2].neurons, neur_attrs, equal=False)
    compare_objs(B[0].neurons, B[2].neurons, neur_attrs, equal=False)


def test_time(Simulator):
    m = nengo.Model('test_time', seed=123)
    sim = Simulator(m)
    sim.run(0.003)
    assert np.allclose(sim.trange(), [0.00, .001, .002])


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
