import numpy as np
import pytest

from nengo.exceptions import ValidationError
import nengo
from nengo.networks.assoc_mem import AssociativeMemory


def similarity(data, target):
    return np.mean(np.dot(data, target.T))


def make_vocab(N, D, rng):
    vocab = None
    while vocab is None or similarity(vocab, vocab) > 0.2:
        vocab = rng.normal(size=(N, D))

        for n in range(N):
            vocab[n, :] = vocab[n, :] / np.linalg.norm(vocab[n, :])

    return vocab


def test_am_basic(Simulator, plt, seed, rng):
    """Basic associative memory test."""

    D = 64
    vocab = np.array([])

    with pytest.raises(ValueError):
        with nengo.Network():
            am = AssociativeMemory(vocab)

    vocab = make_vocab(4, D, rng)

    with nengo.Network("model", seed=seed) as m:
        am = AssociativeMemory(vocab)
        in_node = nengo.Node(output=vocab[0, :], label="input")
        nengo.Connection(in_node, am.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)
        utils_p = nengo.Probe(am.utilities, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()

    plt.subplot(3, 1, 1)
    plt.plot(t, np.dot(sim.data[in_p], vocab.T))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(3, 1, 2)
    plt.plot(t, np.dot(sim.data[out_p], vocab.T))
    plt.plot(t[t > 0.15], np.ones(t.shape)[t > 0.15] * 0.9, c="g", lw=2)
    plt.ylabel("Output")
    plt.subplot(3, 1, 3)
    plt.plot(t, sim.data[utils_p])
    plt.plot(t[t > 0.15], np.ones(t.shape)[t > 0.15] * 0.9, c="g", lw=2)
    plt.ylabel("Utilities")

    assert similarity(sim.data[in_p][t > 0.15], vocab[0, :]) > 0.99
    assert similarity(sim.data[out_p][t > 0.15], vocab[0, :]) > 0.95
    assert similarity(sim.data[utils_p][t > 0.15], np.array([1, 0, 0, 0])) > 0.95
    assert similarity(sim.data[utils_p][t > 0.15], np.array([0, 1, 1, 1])) < 0.001


def test_am_threshold(Simulator, plt, seed, rng):
    """Associative memory thresholding with differing input/output vocabs."""
    D = 64
    vocab = make_vocab(4, D, rng)

    D2 = int(D / 2)
    vocab2 = make_vocab(4, D2, rng)

    def input_func(t):
        return 0.49 * vocab[0, :] if t < 0.1 else 0.8 * vocab[0, :]

    with nengo.Network("model", seed=seed) as m:
        am = AssociativeMemory(vocab, vocab2, threshold=0.5)
        in_node = nengo.Node(output=input_func, label="input")
        nengo.Connection(in_node, am.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)
        utils_p = nengo.Probe(am.utilities, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.3)
    t = sim.trange()
    below_th = t < 0.1
    above_th = t > 0.25

    plt.subplot(2, 1, 1)
    plt.plot(t, np.dot(sim.data[in_p], vocab.T))
    plt.ylabel("Input")
    plt.subplot(2, 1, 2)
    plt.plot(t, np.dot(sim.data[out_p], vocab2.T))
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.9, c="g", lw=2)
    plt.ylabel("Output")

    assert similarity(sim.data[in_p][below_th], vocab[0, :]) > 0.48
    assert similarity(sim.data[in_p][above_th], vocab[0, :]) > 0.79
    assert np.mean(sim.data[out_p][below_th]) < 0.01
    assert similarity(sim.data[out_p][above_th], vocab2[0, :]) > 0.90
    assert similarity(sim.data[utils_p][above_th], np.array([1, 0, 0, 0])) > 0.95
    assert similarity(sim.data[utils_p][above_th], np.array([0, 1, 1, 1])) < 0.001


def test_am_wta(Simulator, plt, seed, rng):
    """Test the winner-take-all ability of the associative memory."""

    D = 64
    vocab = make_vocab(4, D, rng)

    def input_func(t):
        if t < 0.2:
            return vocab[0, :] + 0.8 * vocab[1, :]
        elif t < 0.3:
            return np.zeros(D)
        else:
            return 0.8 * vocab[0, :] + vocab[1, :]

    with nengo.Network("model", seed=seed) as m:
        am = AssociativeMemory(vocab)
        am.add_wta_network()

        in_node = nengo.Node(output=input_func, label="input")
        nengo.Connection(in_node, am.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)
        utils_p = nengo.Probe(am.utilities, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()
    more_a = (t > 0.15) & (t < 0.2)
    more_b = t > 0.45

    plt.subplot(2, 1, 1)
    plt.plot(t, np.dot(sim.data[in_p], vocab.T))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(2, 1, 2)
    plt.plot(t, np.dot(sim.data[out_p], vocab.T))
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.9, c="g", lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.9, c="g", lw=2)
    plt.ylabel("Output")

    assert similarity(sim.data[out_p][more_a], vocab[0, :]) > 0.79
    assert similarity(sim.data[out_p][more_a], vocab[1, :]) < 0.1
    assert similarity(sim.data[out_p][more_b], vocab[1, :]) > 0.79
    assert similarity(sim.data[out_p][more_b], vocab[0, :]) < 0.1
    assert similarity(sim.data[utils_p][more_a], np.array([1, 0, 0, 0])) > 0.95
    assert similarity(sim.data[utils_p][more_a], np.array([0, 1, 1, 1])) < 0.001
    assert similarity(sim.data[utils_p][more_b], np.array([0, 1, 0, 0])) > 0.95
    assert similarity(sim.data[utils_p][more_b], np.array([1, 0, 1, 1])) < 0.001


@pytest.mark.xfail(reason="This test is not consistent", strict=False)
def test_am_complex(Simulator, plt, seed, rng):
    """Complex auto-associative memory test.

    Has a default output vector, outputs utilities, and becomes inhibited.
    """
    D = 64
    vocab = make_vocab(6, D, rng)
    vocab2 = vocab[:4]

    def input_func(t):
        if t < 0.25:
            return 0.6 * vocab[0] + 0.4 * vocab[1]
        elif t < 0.5:
            return 0.4 * vocab[0] + 0.6 * vocab[1]
        else:
            return vocab[4]

    def inhib_func(t):
        return int(t > 0.75)

    with nengo.Network("model", seed=seed) as m:
        am = AssociativeMemory(vocab2, inhibitable=True)
        am.add_default_output_vector(vocab[5])
        am.add_threshold_to_outputs()

        in_node = nengo.Node(output=input_func, label="input")
        inhib_node = nengo.Node(output=inhib_func, label="inhib")
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)
        utils_p = nengo.Probe(am.utilities, synapse=0.05)
        utils_th_p = nengo.Probe(am.thresholded_utilities, synapse=0.05)

    with Simulator(m) as sim:
        sim.run(1.0)
    t = sim.trange()
    # Input: 0.6A + 0.4B
    more_a = (t >= 0.2) & (t < 0.25)
    # Input: 0.4A + 0.6B
    more_b = (t >= 0.45) & (t < 0.5)
    # Input: D (but D isn't in the memory vocabulary, so should output E)
    all_e = (t >= 0.7) & (t < 0.75)
    # Input: D (E) (but inhibited, so should output nothing)
    inhib = t >= 0.95

    def plot(i, y, ylabel):
        plt.subplot(4, 1, i)
        plt.plot(t, y)
        plt.axvline(0.25, c="k")
        plt.axvline(0.5, c="k")
        plt.axvline(0.75, c="k")
        plt.ylabel(ylabel)

    plot(1, np.dot(sim.data[in_p], vocab.T), "Input")
    plot(2, sim.data[utils_p], "Utilities")
    plot(3, sim.data[utils_th_p], "Thresholded utilities")
    plot(4, np.dot(sim.data[out_p], vocab.T), "Output")

    # Check that the output utilities (non-thresholded) are to be expected
    assert all(np.mean(sim.data[utils_p][more_a], axis=0)[:2] > [0.9, 0.35])
    assert all(np.mean(sim.data[utils_p][more_a], axis=0)[2:] < [0.01, 0.01])
    assert all(np.mean(sim.data[utils_p][more_b], axis=0)[:2] > [0.35, 0.9])
    assert all(np.mean(sim.data[utils_p][more_b], axis=0)[2:] < [0.01, 0.01])
    assert similarity(sim.data[utils_p][all_e], np.ones((1, 4))) < 0.05
    assert similarity(sim.data[utils_p][inhib], np.ones((1, 4))) < 0.05

    # Check that the thresholded output utilities are to be expected
    assert all(np.mean(sim.data[utils_th_p][more_a], axis=0)[:2] > [0.9, 0.9])
    assert all(np.mean(sim.data[utils_th_p][more_a], axis=0)[2:] < [0.01, 0.01])
    assert all(np.mean(sim.data[utils_th_p][more_b], axis=0)[:2] > [0.9, 0.9])
    assert all(np.mean(sim.data[utils_th_p][more_b], axis=0)[2:] < [0.01, 0.01])
    assert similarity(sim.data[utils_th_p][all_e], np.ones((1, 4))) < 0.05
    assert similarity(sim.data[utils_th_p][inhib], np.ones((1, 4))) < 0.05

    # Check that the output values are to be expected
    assert similarity(sim.data[out_p][more_a], vocab[0]) > 0.7
    assert similarity(sim.data[out_p][more_a], vocab[1]) > 0.7
    assert similarity(sim.data[out_p][more_b], vocab[0]) > 0.7
    assert similarity(sim.data[out_p][more_b], vocab[1]) > 0.7
    assert similarity(sim.data[out_p][all_e], vocab[5]) > 0.7
    assert similarity(sim.data[out_p][inhib], np.ones((1, D))) < 0.05


def test_repeat_config_warning():
    """tests a warning is run on repeat config"""
    with nengo.Network():
        test_am = AssociativeMemory([0])

    test_am.add_threshold_to_outputs()
    with pytest.warns(UserWarning, match="already configured with thresholded outputs"):
        test_am.add_threshold_to_outputs()

    test_am.add_wta_network()
    with pytest.warns(UserWarning, match="already configured with a WTA network"):
        test_am.add_wta_network()


def test_add_output_mapping(rng):
    """tests add_output_mapping edge cases and errors"""
    vocab = make_vocab(4, 64, rng)

    with nengo.Network():
        test_am = AssociativeMemory(vocab)

    output_vectors = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    test_am.add_output_mapping("test", output_vectors)
    assert isinstance(test_am.test, nengo.Node)

    with pytest.raises(ValidationError, match="Name .* already exists as a node"):
        test_am.add_output_mapping("test", output_vectors)


def test_add_input_mapping(rng):
    """tests add_input_mapping edge cases and errors"""
    vocab = make_vocab(4, 64, rng)

    with nengo.Network():
        test_am = AssociativeMemory(vocab)

    input_vectors = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    test_am.add_input_mapping("test", input_vectors, input_scales=[1, 2, 3, 4])
    assert isinstance(test_am.test, nengo.Node)

    with pytest.raises(ValidationError, match="Name .* already exists as a node"):
        test_am.add_input_mapping("test", input_vectors, input_scales=[1, 2, 3, 4])

    # wrong input scales shape
    with pytest.raises(ValidationError, match="Number of input_scale values"):
        test_am.add_input_mapping("test2", input_vectors, input_scales=[1])


def test_associativememory_errors(rng):
    """tests multiple errors in AssociativeMemory"""
    vocab = make_vocab(4, 64, rng)

    with nengo.Network():
        with pytest.raises(
            ValidationError, match="Number of input vectors.*cannot be 0"
        ):
            AssociativeMemory(np.zeros((0, 1)))

        with pytest.raises(
            ValidationError,
            match="Number of input vectors.*does not match number of output vectors",
        ):
            AssociativeMemory(vocab, output_vectors=np.zeros((1, 64)))

        with pytest.raises(
            ValidationError,
            match="Number of threshold values.*does not match number of input vectors",
        ):
            AssociativeMemory(vocab, threshold=[1])


def test_associativememory_edge_cases(seed, rng):
    """Tests that edge case code runs without error

    TODO: In the future, these features should be tested in an integration test.
    """
    vocab = make_vocab(4, 64, rng)
    out_vectors = rng.uniform(-1, 1, size=(4, 3))

    with nengo.Network(seed=seed):
        # test that an iterable threshold works
        am = AssociativeMemory(vocab, threshold=[0.1, 0.2, 0.3, 0.4])

        am.add_threshold_to_outputs()

        # test add_output_mapping works when `thresh_ens is not None`
        am.add_output_mapping("test", out_vectors)
        inp, out = am.thresh_ens.output, am.test
        conn = [c for c in am.out_conns if c.pre is inp and c.post is out][0]
        assert np.allclose(conn.transform.init, out_vectors.T)

        # test add_default_output_vector works when `thresh_ens is not None`
        am.add_default_output_vector(np.ones(64))
        assert len(am.default_vector_inhibit_conns) == 1
        conn = am.default_vector_inhibit_conns[0]
        assert conn.pre is am.thresh_ens.output
