import pytest

import nengo
import nengo.builder

node_attrs = ('output',)
ens_attrs = ('label', 'dimensions', 'radius')
connection_attrs = ('filter', 'transform')


def compare(orig, copy):
    if isinstance(orig, nengo.Node):
        attrs = node_attrs
    elif isinstance(orig, nengo.Ensemble):
        attrs = ens_attrs
    elif isinstance(orig, nengo.Connection):
        attrs = connection_attrs

    for attr in attrs:
        assert getattr(orig, attr) == getattr(copy, attr)
    for p_o, p_c in zip(orig.probes.values(), copy.probes.values()):
        assert len(p_o) == len(p_c)


def mybuilder(model, dt):
    model.dt = dt
    model.seed = 0
    if not hasattr(model, 'probes'):
        model.probes = []
    return model


def test_build():
    m = nengo.Model('test_build', seed=123)
    input = nengo.Node(output=1)
    A = nengo.Ensemble(nengo.LIF(40), 1)
    B = nengo.Ensemble(nengo.LIF(20), 1)
    nengo.Connection(input, A)
    nengo.Connection(A, B, function=lambda x: x ** 2)
    # input_p = nengo.Probe(input, 'output')
    # A_p = nengo.Probe(A, 'decoded_output', filter=0.01)
    # B_p = nengo.Probe(B, 'decoded_output', filter=0.01)

    mcopy = nengo.Simulator(m).model
    assert [o.label for o in m.objs] == [o.label for o in mcopy.objs]

    for o, copy_o in zip(m.objs, mcopy.objs):
        compare(o, copy_o)
    for c, copy_c in zip(m.connections, mcopy.connections):
        compare(c, copy_c)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
