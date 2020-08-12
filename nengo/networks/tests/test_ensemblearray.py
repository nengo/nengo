import numpy as np
import pytest

import nengo
from nengo.dists import Choice, UniformHypersphere
from nengo.exceptions import ValidationError


def test_multidim(Simulator, plt, seed, rng, allclose):
    """Tests with multiple dimensions per ensemble"""
    dims = 3
    n_neurons = 60
    radius = 1.0

    a = rng.uniform(low=-0.7, high=0.7, size=dims)
    b = rng.uniform(low=-0.7, high=0.7, size=dims)
    c = np.zeros(2 * dims)
    c[::2] = a
    c[1::2] = b

    model = nengo.Network(seed=seed)
    with model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        A = nengo.networks.EnsembleArray(n_neurons, dims, radius=radius)
        B = nengo.networks.EnsembleArray(n_neurons, dims, radius=radius)
        C = nengo.networks.EnsembleArray(
            n_neurons * 2, dims, ens_dimensions=2, radius=radius
        )
        nengo.Connection(inputA, A.input)
        nengo.Connection(inputB, B.input)
        nengo.Connection(A.output, C.input[::2])
        nengo.Connection(B.output, C.input[1::2])

        A_p = nengo.Probe(A.output, synapse=0.03)
        B_p = nengo.Probe(B.output, synapse=0.03)
        C_p = nengo.Probe(C.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.4)

    t = sim.trange()

    def plot(sim, a, p, title=""):
        a_ref = np.tile(a, (len(t), 1))
        a_sim = sim.data[p]
        colors = ["b", "g", "r", "c", "m", "y"]
        for i in range(a_sim.shape[1]):
            plt.plot(t, a_ref[:, i], "--", color=colors[i % 6])
            plt.plot(t, a_sim[:, i], "-", color=colors[i % 6])
        plt.xticks(np.linspace(0, 0.4, 5))
        plt.xlim(right=t[-1])
        plt.title(title)

    plt.subplot(131)
    plot(sim, a, A_p, title="A")
    plt.subplot(132)
    plot(sim, b, B_p, title="B")
    plt.subplot(133)
    plot(sim, c, C_p, title="C")

    a_sim = sim.data[A_p][t > 0.2].mean(axis=0)
    b_sim = sim.data[B_p][t > 0.2].mean(axis=0)
    c_sim = sim.data[C_p][t > 0.2].mean(axis=0)

    rtol, atol = 0.1, 0.05
    assert allclose(a, a_sim, atol=atol, rtol=rtol)
    assert allclose(b, b_sim, atol=atol, rtol=rtol)
    assert allclose(c, c_sim, atol=atol, rtol=rtol)


def _mmul_transforms(A_shape, B_shape, C_dim):
    transformA = np.zeros((C_dim, A_shape[0] * A_shape[1]))
    transformB = np.zeros((C_dim, B_shape[0] * B_shape[1]))

    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            for k in range(B_shape[1]):
                tmp = j + k * A_shape[1] + i * B_shape[0] * B_shape[1]
                transformA[tmp * 2][j + i * A_shape[1]] = 1
                transformB[tmp * 2 + 1][k + j * B_shape[1]] = 1

    return transformA, transformB


def test_multifunc(Simulator, plt, seed, rng):
    """Tests with different functions computed by each ensemble."""
    dims = 3
    n_neurons = 60

    inp = rng.uniform(low=-0.7, high=0.7, size=dims)
    functions = [lambda x: [x * 2], lambda x: [-x, -x * 2], lambda x: [0.5 * x]]
    output = []
    for i, func in enumerate(functions):
        output.extend(func(inp[i]))

    model = nengo.Network(seed=seed)
    with model:
        inp_node = nengo.Node(inp)
        ea = nengo.networks.EnsembleArray(n_neurons, dims)
        ea_funcs = ea.add_output("multiple functions", function=functions)

        nengo.Connection(inp_node, ea.input)

        ea_p = nengo.Probe(ea.output, synapse=0.03)
        ea_funcs_p = nengo.Probe(ea_funcs, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.4)

    t = sim.trange()

    def plot(sim, expected, probe, title=""):
        simdata = sim.data[probe]
        colors = ["b", "g", "r", "c"]
        for i in range(simdata.shape[1]):
            plt.axhline(expected[i], ls="--", color=colors[i % 4])
            plt.plot(t, simdata[:, i], color=colors[i % 4])
        plt.xticks(np.linspace(0, 0.4, 5))
        plt.xlim(right=t[-1])
        plt.title(title)

    plt.subplot(121)
    plot(sim, inp, ea_p, title="A")
    plt.subplot(122)
    plot(sim, output, ea_funcs_p, title="B")


def test_matrix_mul(Simulator, plt, seed, allclose):
    N = 100

    Amat = np.asarray([[0.5, -0.5]])
    Bmat = np.asarray([[0.8, 0.3], [0.2, 0.7]])
    radius = 1

    model = nengo.Network(label="Matrix Multiplication", seed=seed)
    with model:
        A = nengo.networks.EnsembleArray(N, Amat.size, radius=radius, label="A")
        B = nengo.networks.EnsembleArray(N, Bmat.size, radius=radius, label="B")

        inputA = nengo.Node(output=Amat.ravel())
        inputB = nengo.Node(output=Bmat.ravel())
        nengo.Connection(inputA, A.input)
        nengo.Connection(inputB, B.input)
        A_p = nengo.Probe(A.output, synapse=0.03)
        B_p = nengo.Probe(B.output, synapse=0.03)

        Cdims = Amat.size * Bmat.shape[1]
        C = nengo.networks.EnsembleArray(
            N,
            Cdims,
            ens_dimensions=2,
            radius=np.sqrt(2) * radius,
            encoders=Choice([[1, 1], [-1, 1], [1, -1], [-1, -1]]),
        )

        transformA, transformB = _mmul_transforms(Amat.shape, Bmat.shape, C.dimensions)

        nengo.Connection(A.output, C.input, transform=transformA)
        nengo.Connection(B.output, C.input, transform=transformB)

        D = nengo.networks.EnsembleArray(
            N, Amat.shape[0] * Bmat.shape[1], radius=radius
        )

        transformC = np.zeros((D.dimensions, Bmat.size))
        for i in range(Bmat.size):
            transformC[i // Bmat.shape[0]][i] = 1

        prod = C.add_output("product", lambda x: x[0] * x[1])

        nengo.Connection(prod, D.input, transform=transformC)
        D_p = nengo.Probe(D.output, synapse=0.03)

    with Simulator(model) as sim:
        sim.run(0.3)

    t = sim.trange()
    tmask = t >= 0.2

    plt.plot(t, sim.data[D_p])
    for d in np.dot(Amat, Bmat).flatten():
        plt.axhline(d, color="k")

    tols = dict(atol=0.1, rtol=0.01)
    for i in range(Amat.size):
        assert allclose(sim.data[A_p][tmask, i], Amat.flat[i], **tols)
    for i in range(Bmat.size):
        assert allclose(sim.data[B_p][tmask, i], Bmat.flat[i], **tols)

    Dmat = np.dot(Amat, Bmat)
    for i in range(Amat.shape[0]):
        for k in range(Bmat.shape[1]):
            data_ik = sim.data[D_p][tmask, i * Bmat.shape[1] + k]
            assert allclose(data_ik, Dmat[i, k], **tols)


def test_arguments():
    """Make sure EnsembleArray accepts the right arguments."""
    with pytest.raises(ValidationError):
        nengo.networks.EnsembleArray(nengo.LIF(10), 1, dimensions=2)


def test_directmode_errors():
    with nengo.Network() as net:
        net.config[nengo.Ensemble].neuron_type = nengo.Direct()

        ea = nengo.networks.EnsembleArray(10, 2)
        with pytest.raises(ValidationError):
            ea.add_neuron_input()
        with pytest.raises(ValidationError):
            ea.add_neuron_output()


def test_neuroninput(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        inp = nengo.Node([-10] * 20)
        ea = nengo.networks.EnsembleArray(10, 2)
        ea.add_neuron_input()
        nengo.Connection(inp, ea.neuron_input, synapse=None)
        p = nengo.Probe(ea.output)

    with Simulator(net) as sim:
        sim.run(0.01)
    assert np.all(sim.data[p] < 1e-2) and np.all(sim.data[p] > -1e-2)


def test_neuronoutput(Simulator, seed):
    with nengo.Network(seed=seed) as net:
        ea = nengo.networks.EnsembleArray(10, 2, encoders=nengo.dists.Choice([[1]]))
        ea.add_neuron_output()
        inp = nengo.Node([-10, -10])
        nengo.Connection(inp, ea.input, synapse=None)
        p = nengo.Probe(ea.neuron_output)

    with Simulator(net) as sim:
        sim.run(0.01)
    assert np.all(sim.data[p] < 1e-2)


def test_ndarrays(Simulator, rng, allclose):
    encoders = UniformHypersphere(surface=True).sample(10, 1, rng=rng)
    max_rates = rng.uniform(200, 400, size=10)
    intercepts = rng.uniform(-1, 1, size=10)
    eval_points = rng.uniform(-1, 1, size=(100, 1))
    with nengo.Network() as net:
        ea = nengo.networks.EnsembleArray(
            10,
            2,
            encoders=encoders,
            max_rates=max_rates,
            intercepts=intercepts,
            eval_points=eval_points,
            n_eval_points=eval_points.shape[0],
        )

    with Simulator(net) as sim:
        pass
    built = sim.data[ea.ea_ensembles[0]]
    assert allclose(built.encoders, encoders)
    assert allclose(built.max_rates, max_rates)
    assert allclose(built.intercepts, intercepts)
    assert allclose(built.eval_points, eval_points)
    assert built.eval_points.shape == eval_points.shape

    with nengo.Network() as net:
        # Incorrect shapes don't fail until build
        ea = nengo.networks.EnsembleArray(
            10,
            2,
            encoders=encoders,
            max_rates=max_rates,
            intercepts=intercepts,
            eval_points=eval_points[:10],
            n_eval_points=eval_points.shape[0],
        )

    with pytest.raises(ValidationError):
        with Simulator(net) as sim:
            pass


def test_add_input_output_errors():
    """Ensures warnings and errors are thrown as appropriate"""
    with nengo.Network():
        A = nengo.networks.EnsembleArray(n_neurons=10, n_ensembles=3)

    A.add_neuron_input()
    with pytest.warns(UserWarning, match="neuron_input already exists"):
        A.add_neuron_input()

    A.add_neuron_output()
    with pytest.warns(UserWarning, match="neuron_output already exists"):
        A.add_neuron_output()

    A.add_output("my_output", np.sin)
    with pytest.raises(ValidationError, match="Cannot add output.*already an attr"):
        A.add_output("my_output", np.sin)
    with pytest.raises(ValidationError, match="Cannot add output.*already an attr"):
        A.add_output("neuron_input", np.sin)

    with pytest.raises(ValidationError, match="Must have one function per ensemble"):
        A.add_output("test", [np.sin] * (A.n_ensembles + 1))

    with pytest.raises(ValidationError, match="'function' must be a callable"):
        A.add_output("test", "not a function")
