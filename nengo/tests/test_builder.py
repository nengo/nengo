import numpy as np
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


def test_pyfunc():
    """Test Python Function nonlinearity"""
    dt = 0.001
    d = 3
    n_steps = 3
    n_trials = 3

    rng = np.random.RandomState(seed=987)

    for i in range(n_trials):
        A = rng.normal(size=(d, d))
        fn = lambda t, x: np.cos(np.dot(A, x))

        x = np.random.normal(size=d)

        m = nengo.Model("")
        ins = nengo.builder.Signal(x, name='ins')
        pop = nengo.builder.PythonFunction(fn=fn, n_in=d, n_out=d)
        m.operators = []
        b = nengo.builder.Builder()
        b.model = m
        b.build_pyfunc(pop)
        m.operators += [
            nengo.builder.DotInc(
                nengo.builder.Signal(np.eye(d)), ins, pop.input_signal),
            nengo.builder.ProdUpdate(nengo.builder.Signal(np.eye(d)),
                                     pop.output_signal,
                                     nengo.builder.Signal(0),
                                     ins)
        ]

        sim = nengo.Simulator(m, dt=dt, builder=mybuilder)

        p0 = np.zeros(d)
        s0 = np.array(x)
        for j in range(n_steps):
            tmp = p0
            p0 = fn(0, s0)
            s0 = tmp
            sim.step()
            assert np.allclose(s0, sim.signals[ins])
            assert np.allclose(p0, sim.signals[pop.output_signal])


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
