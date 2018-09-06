import numpy as np
import pytest

import nengo
from nengo.exceptions import SimulationError, ValidationError
from nengo.rc import rc


def test_time(Simulator, allclose):
    with nengo.Network() as model:
        u = nengo.Node(output=lambda t: t)
        up = nengo.Probe(u)

    with Simulator(model) as sim:
        sim.run(1.0)

    t = sim.trange()
    x = sim.data[up].flatten()
    assert allclose(t, x, atol=1e-7, rtol=1e-4)


def test_simple(Simulator, plt, seed, allclose):
    m = nengo.Network(seed=seed)
    with m:
        input = nengo.Node(output=np.sin)
        p = nengo.Probe(input, "output")

    with Simulator(m) as sim:
        sim.run(0.5)

    plt.plot(sim.trange(), sim.data[p], label="sin")
    plt.legend(loc="best")

    sim_t = sim.trange()
    sim_in = sim.data[p].ravel()
    assert allclose(sim_in, np.sin(sim_t))


def test_connected(Simulator, plt, seed, allclose):
    m = nengo.Network(seed=seed)
    with m:
        input = nengo.Node(output=np.sin, label="input")
        output = nengo.Node(output=lambda t, x: np.square(x), size_in=1, label="output")
        nengo.Connection(input, output, synapse=None)  # Direct connection
        p_in = nengo.Probe(input, "output")
        p_out = nengo.Probe(output, "output")

    with Simulator(m) as sim:
        sim.run(0.5)

    t = sim.trange()
    plt.plot(t, sim.data[p_in], label="sin")
    plt.plot(t, sim.data[p_out], label="sin squared")
    plt.plot(t, np.sin(t), label="ideal sin")
    plt.plot(t, np.sin(t) ** 2, label="ideal squared")
    plt.legend(loc="best")

    sim_t = sim.trange()
    sim_sin = sim.data[p_in].ravel()
    sim_sq = sim.data[p_out].ravel()
    assert allclose(sim_sin, np.sin(sim_t))
    assert allclose(sim_sq, sim_sin ** 2)


def test_passthrough(Simulator, plt, seed, allclose):
    m = nengo.Network(seed=seed)
    with m:
        in1 = nengo.Node(output=np.sin)
        in2 = nengo.Node(output=lambda t: t)
        passthrough = nengo.Node(size_in=1)
        out = nengo.Node(output=lambda t, x: x, size_in=1)

        nengo.Connection(in1, passthrough, synapse=None)
        nengo.Connection(in2, passthrough, synapse=None)
        nengo.Connection(passthrough, out, synapse=None)

        in1_p = nengo.Probe(in1, "output")
        in2_p = nengo.Probe(in2, "output")
        out_p = nengo.Probe(out, "output")

    with Simulator(m) as sim:
        sim.run(0.5)

    plt.plot(sim.trange(), sim.data[in1_p] + sim.data[in2_p], label="in+in2")
    plt.plot(sim.trange()[:-2], sim.data[out_p][2:], label="out")
    plt.legend(loc="best")

    sim_in = sim.data[in1_p] + sim.data[in2_p]
    sim_out = sim.data[out_p]
    assert allclose(sim_in, sim_out)


def test_passthrough_filter(Simulator, plt, seed, allclose):
    m = nengo.Network(seed=seed)
    with m:
        omega = 2 * np.pi * 5
        u = nengo.Node(output=lambda t: np.sin(omega * t))
        passthrough = nengo.Node(size_in=1)
        v = nengo.Node(output=lambda t, x: x, size_in=1)

        synapse = nengo.Lowpass(0.3)
        nengo.Connection(u, passthrough, synapse=None)
        nengo.Connection(passthrough, v, synapse=synapse)

        up = nengo.Probe(u)
        vp = nengo.Probe(v)

    with Simulator(m) as sim:
        sim.run(1.0)

    t = sim.trange()
    x = sim.data[up]
    y = synapse.filt(x, dt=sim.dt, y0=0)
    z = sim.data[vp]

    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, z)

    assert allclose(y[:-1], z[1:], atol=1e-7, rtol=1e-4)


def test_circular(Simulator, seed, allclose):
    m = nengo.Network(seed=seed)
    with m:
        a = nengo.Node(output=lambda t, x: x + 1, size_in=1)
        b = nengo.Node(output=lambda t, x: x + 1, size_in=1)
        nengo.Connection(a, b, synapse=0)
        nengo.Connection(b, a, synapse=0)

        a_p = nengo.Probe(a, "output")
        b_p = nengo.Probe(b, "output")

    with Simulator(m) as sim:
        sim.run(0.5)

    assert allclose(sim.data[a_p], sim.data[b_p])


def test_outputparam_errors(Simulator):
    with nengo.Network() as model:
        # valid values
        nengo.Node(output=lambda t: t + 1)
        nengo.Node(output=0)
        nengo.Node(output=[0, 1])
        nengo.Node(output=nengo.processes.WhiteNoise())
        nengo.Node(size_in=1)

        # type errors
        with pytest.raises(ValidationError):
            nengo.Node(output=object())

        # function errors
        nengo.Node(output=lambda t, x=[0]: t + 1, size_in=1)
        with pytest.raises(ValidationError):
            nengo.Node(output=lambda t, x: x + 1)
        with pytest.raises(ValidationError):
            nengo.Node(output=lambda t: t + 1, size_in=1)
        with pytest.raises(ValidationError):
            nengo.Node(output=lambda t, x, y: t + 1, size_in=2)
        with pytest.raises(ValidationError):
            nengo.Node(output=[0], size_in=1)
        with pytest.raises(ValidationError):
            nengo.Node(output=0, size_in=1)

        # shape errors
        with pytest.raises(ValidationError):
            nengo.Node(output=[[1, 2], [3, 4]])
        with pytest.raises(ValidationError):
            nengo.Node(output=lambda t: [[t, t + 1]])
        with pytest.raises(ValidationError):
            nengo.Node(output=[[3, 1], [2, 9]], size_out=4)
        with pytest.raises(ValidationError):
            nengo.Node(output=[1, 2, 3, 4, 5], size_out=4)

    with Simulator(model):
        pass


def test_none(Simulator, seed):
    """Test for nodes that output None."""

    model = nengo.Network(seed=seed)

    # This function will fail, because at build time it will be
    # detected as producing output (func is called with 0 input)
    # but during the run it will produce None when t >=0.5
    def input_function(t):
        if t < 0.005:
            return [1]

    with model:
        u = nengo.Node(output=input_function)
        a = nengo.Ensemble(10, dimensions=1)
        nengo.Connection(u, a)

    with Simulator(model) as sim:
        with pytest.raises(SimulationError):
            sim.run(0.01)

    # This function will pass (with a warning), because it will
    # be determined at run time that the output function
    # returns None
    def none_function(t):
        pass

    model2 = nengo.Network()
    with model2:
        nengo.Node(output=none_function)

    with Simulator(model2) as sim:
        sim.run(0.01)


def test_unconnected_node(Simulator):
    """Make sure unconnected nodes still run."""
    hits = np.array(0)
    dt = 0.001

    def f(t):
        hits[...] += 1

    model = nengo.Network()
    with model:
        nengo.Node(f, size_in=0, size_out=0)
    with Simulator(model) as sim:
        assert hits == 0
        sim.run(dt)
        assert hits == 1
        sim.run(dt)
        assert hits == 2


def test_len():
    with nengo.Network():
        n1 = nengo.Node(None, size_in=1)
        n3 = nengo.Node([1, 2, 3])
        n4 = nengo.Node(lambda t: np.arange(4) * t)

    assert len(n1) == 1
    assert len(n3) == 3
    assert len(n4) == 4
    assert len(n1[0]) == 1
    assert len(n4[1:3]) == 2


def test_set_arraylike_output(Simulator):
    with nengo.Network() as model:
        # if output is None, size_out == size_in
        with pytest.warns(UserWarning):
            # warns since size_in != size_out and output is None
            passthrough = nengo.Node(None, size_in=20, size_out=30)
        assert passthrough.output is None
        assert passthrough.size_out == 20

        # if output is an array-like...
        # size_in must be 0
        with pytest.raises(ValidationError):
            nengo.Node(np.ones(1), size_in=1)
        # size_out must match
        with pytest.raises(ValidationError):
            nengo.Node(np.ones(3), size_out=2)
        # must be scalar or vector, not matrix
        with pytest.raises(ValidationError):
            nengo.Node(np.ones((2, 2)))
        # scalar gets promoted to float vector
        scalar = nengo.Node(2)
        assert scalar.output.shape == (1,)
        assert scalar.output.dtype == rc.float_dtype
        # vector stays 1D
        vector = nengo.Node(np.arange(3))
        assert vector.output.shape == (3,)
        assert vector.output.dtype == rc.float_dtype

    with Simulator(model):  # Ensure it all builds
        pass


def test_set_callable_output(Simulator):
    model = nengo.Network()

    counter = []

    def accumulate(t):
        counter.append(t)
        return t

    with model:
        # if we pass size_out, function should not be called
        assert len(counter) == 0
        accum_func = nengo.Node(accumulate, size_out=1)
        assert len(counter) == 0
        assert accum_func.size_out == 1

    def noreturn(t):
        pass

    with model:
        # if the function returns None, size_out == 0
        noreturn_func = nengo.Node(noreturn)
        assert noreturn_func.size_out == 0

    class TestObject:
        def __init__(self):
            self.val = 0

        def step(self, t):
            return self.val

    with model:
        # passing methods into nodes shouldn't cause validation errors
        test_obj = TestObject()
        nengo.Node(test_obj.step, size_out=1)

    with model:
        with pytest.raises(ValidationError):
            nengo.Node(lambda: 2.0, size_out=1)
        # if size_in is 0, should only take in t
        with pytest.raises(ValidationError):
            nengo.Node(lambda t, x: 2.0, size_in=0)
        with pytest.raises(ValidationError):
            nengo.Node(lambda t, x: 2.0, size_in=0, size_out=0)
        # if size_in > 0, should take both t and x
        with pytest.raises(ValidationError):
            nengo.Node(lambda t: t ** 2, size_in=1)
        # function must return a scalar or vector, not matrix
        with pytest.raises(ValidationError):
            nengo.Node(lambda t: np.ones((2, 2)))
        # variable length argument lists should be allowed (used in Nengo SPA)
        nengo.Node(lambda *args: [2.0], size_in=0, size_out=1)
        nengo.Node(lambda *args: [2.0], size_in=1, size_out=1)

    with Simulator(model):  # Ensure it all builds
        pass


def test_delay(Simulator, plt):
    with nengo.Network() as model:
        a = nengo.Node(output=np.sin)
        b = nengo.Node(output=lambda t, x: -x, size_in=1)
        nengo.Connection(a[[0]], b, synapse=None)

        ap = nengo.Probe(a)
        bp = nengo.Probe(b)

    with Simulator(model) as sim:
        sim.run(0.005)

    plt.plot(sim.trange(), sim.data[ap])
    plt.plot(sim.trange(), -sim.data[bp])


def test_args(Simulator):
    class Fn:
        def __init__(self):
            self.last_x = None

        def __call__(self, t, x):
            assert isinstance(t, float)
            assert isinstance(x, np.ndarray)
            assert self.last_x is not x  # x should be a new copy on each call
            self.last_x = x
            assert x[0] == t

    with nengo.Network() as model:
        u = nengo.Node(lambda t: t)
        v = nengo.Node(Fn(), size_in=1, size_out=0)
        nengo.Connection(u, v, synapse=None)

    with Simulator(model) as sim:
        sim.run(0.01)


def test_wrong_output():
    """Setting a node as an input used to cause unbounded memory allocation."""

    with nengo.Network():
        node1 = nengo.Node(output=lambda t: t)

        with pytest.raises(ValueError):
            nengo.Node(node1)


def test_seed_error():
    """Setting a Node seed is currently not implemented."""
    with nengo.Network():
        with pytest.raises(NotImplementedError):
            nengo.Node(seed=1)


def test_node_with_offset_array_view(Simulator):
    v = np.array([[1.0, 2.0], [3.0, 4.0]])
    with nengo.Network() as model:
        node = nengo.Node(v[1])
        probe = nengo.Probe(node)
        assert probe

    with Simulator(model):
        pass


def test_node_with_unusual_strided_view(Simulator, seed):
    v = np.array([1.0, 2.0], dtype=complex)  # 16 byte itemsize
    with nengo.Network(seed=seed) as model:
        node = nengo.Node(v.real)  # 8 byte itemsize, but 16 byte strides
        probe = nengo.Probe(node)
        assert probe

    with Simulator(model):
        pass


@pytest.mark.parametrize("badval", [np.inf, np.nan, "string"])
def test_invalid_values(Simulator, badval):
    with nengo.Network() as model:
        with pytest.raises(ValidationError):
            node = nengo.Node(badval)

    with nengo.Network() as model:
        node = nengo.Node(lambda t: badval)
        ens = nengo.Ensemble(10, 1)
        nengo.Connection(node, ens)

    with Simulator(model) as sim:
        with pytest.raises(SimulationError):
            sim.run(0.01)
