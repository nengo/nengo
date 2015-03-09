import numpy as np

import nengo
from nengo.spa import Vocabulary
from nengo.spa.assoc_mem import AssociativeMemory


def similarity(data, target):
    return np.mean(np.dot(data, target.T))


def test_am_basic(Simulator, plt, seed, rng):
    """Basic associative memory test."""

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    with nengo.Network('model', seed=seed) as m:
        am = AssociativeMemory(vocab)
        in_node = nengo.Node(output=vocab.parse("A").v, label='input')
        nengo.Connection(in_node, am.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)

    sim = Simulator(m)
    sim.run(0.2)
    t = sim.trange()

    plt.subplot(2, 1, 1)
    plt.plot(t, nengo.spa.similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.legend(vocab.keys, loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(t, nengo.spa.similarity(sim.data[out_p], vocab))
    plt.plot(t[t > 0.15], np.ones(t.shape)[t > 0.15] * 0.8, c='g', lw=2)
    plt.ylabel("Output")
    plt.legend(vocab.keys, loc='best')

    assert similarity(sim.data[in_p][t > 0.15], vocab.parse("A").v) > 0.99
    assert similarity(sim.data[out_p][t > 0.15], vocab.parse("A").v) > 0.8


def test_am_threshold(Simulator, plt, seed, rng):
    """Associative memory thresholding with differing input/output vocabs."""
    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    D2 = int(D / 2)
    vocab2 = Vocabulary(D2, rng=rng)
    vocab2.parse('A+B+C+D')

    def input_func(t):
        return vocab.parse('0.49*A').v if t < 0.1 else vocab.parse('0.79*A').v

    with nengo.Network('model', seed=seed) as m:
        am = AssociativeMemory(vocab, vocab2, threshold=0.5)
        in_node = nengo.Node(output=input_func, label='input')
        nengo.Connection(in_node, am.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)

    sim = Simulator(m)
    sim.run(0.3)
    t = sim.trange()
    below_th = t < 0.1
    above_th = t > 0.25

    plt.subplot(2, 1, 1)
    plt.plot(t, nengo.spa.similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.legend(vocab.keys, loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(t, nengo.spa.similarity(sim.data[out_p], vocab2))
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.8, c='g', lw=2)
    plt.ylabel("Output")
    plt.legend(vocab.keys, loc='best')

    assert similarity(sim.data[in_p][below_th], vocab.parse("A").v) > 0.48
    assert similarity(sim.data[in_p][above_th], vocab.parse("A").v) > 0.78
    assert similarity(sim.data[out_p][below_th], vocab2.parse("0").v) < 0.01
    assert similarity(sim.data[out_p][above_th], vocab2.parse("A").v) > 0.8


def test_am_wta(Simulator, plt, seed, rng):
    """Test the winner-take-all ability of the associative memory."""

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    def input_func(t):
        if t < 0.2:
            return vocab.parse('A+0.8*B').v
        elif t < 0.3:
            return np.zeros(D)
        else:
            return vocab.parse('0.8*A+B').v

    with nengo.Network('model', seed=seed) as m:
        am = AssociativeMemory(vocab, wta_output=True)
        in_node = nengo.Node(output=input_func, label='input')
        nengo.Connection(in_node, am.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)

    sim = Simulator(m)
    sim.run(0.5)
    t = sim.trange()
    more_a = (t > 0.15) & (t < 0.2)
    more_b = t > 0.45

    plt.subplot(2, 1, 1)
    plt.plot(t, nengo.spa.similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.legend(vocab.keys, loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(t, nengo.spa.similarity(sim.data[out_p], vocab))
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.8, c='g', lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.8, c='g', lw=2)
    plt.ylabel("Output")
    plt.legend(vocab.keys, loc='best')

    assert similarity(sim.data[out_p][more_a], vocab.parse("A").v) > 0.8
    assert similarity(sim.data[out_p][more_a], vocab.parse("B").v) < 0.2
    assert similarity(sim.data[out_p][more_b], vocab.parse("B").v) > 0.8
    assert similarity(sim.data[out_p][more_b], vocab.parse("A").v) < 0.2


def test_am_complex(Simulator, plt, seed, rng):
    """Complex auto-associative memory test.

    Has a default output vector, outputs utilities, and becomes inhibited.
    """
    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D+E+F')

    vocab2 = vocab.create_subset(["A", "B", "C", "D"])

    def input_func(t):
        if t < 0.25:
            return vocab.parse('A+0.8*B').v
        elif t < 0.5:
            return vocab.parse('0.8*A+B').v
        else:
            return vocab.parse('E').v

    def inhib_func(t):
        return int(t > 0.75)

    with nengo.Network('model', seed=seed) as m:
        am = AssociativeMemory(vocab2,
                               default_output_vector=vocab.parse("F").v,
                               inhibitable=True,
                               output_utilities=True,
                               output_thresholded_utilities=True)
        in_node = nengo.Node(output=input_func, label='input')
        inhib_node = nengo.Node(output=inhib_func, label='inhib')
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(am.output, synapse=0.03)
        utils_p = nengo.Probe(am.utilities, synapse=0.05)
        utils_th_p = nengo.Probe(am.thresholded_utilities, synapse=0.05)

    sim = Simulator(m)
    sim.run(1.0)
    t = sim.trange()
    # Input: A+0.8B
    more_a = (t >= 0.2) & (t < 0.25)
    # Input: 0.8B+A
    more_b = (t >= 0.45) & (t < 0.5)
    # Input: E (but E isn't in the memory vocabulary, so should output F)
    all_e = (t >= 0.7) & (t < 0.75)
    # Input: E (but inhibited, so should output nothing)
    inhib = (t >= 0.95)

    def plot(i, y, ylabel):
        plt.subplot(4, 1, i)
        plt.plot(t, y)
        plt.axvline(0.25, c='k')
        plt.axvline(0.5, c='k')
        plt.axvline(0.75, c='k')
        plt.ylabel(ylabel)
        plt.legend(vocab.keys[:y.shape[1]], loc='best', fontsize='xx-small')
    plot(1, nengo.spa.similarity(sim.data[in_p], vocab), "Input")
    plot(2, sim.data[utils_p], "Utilities")
    plot(3, sim.data[utils_th_p], "Thresholded utilities")
    plot(4, nengo.spa.similarity(sim.data[out_p], vocab), "Output")

    assert all(np.mean(sim.data[utils_p][more_a], axis=0)[:2] > [0.8, 0.5])
    assert all(np.mean(sim.data[utils_p][more_a], axis=0)[2:] < [0.01, 0.01])
    assert all(np.mean(sim.data[utils_p][more_b], axis=0)[:2] > [0.5, 0.8])
    assert all(np.mean(sim.data[utils_p][more_b], axis=0)[2:] < [0.01, 0.01])
    assert similarity(sim.data[utils_p][all_e], np.ones((1, 4))) < 0.05
    assert similarity(sim.data[utils_p][inhib], np.ones((1, 4))) < 0.05
    assert all(np.mean(sim.data[utils_th_p][more_a], axis=0)[:2] > [0.8, 0.8])
    assert all(
        np.mean(sim.data[utils_th_p][more_a], axis=0)[2:] < [0.01, 0.01])
    assert all(np.mean(sim.data[utils_th_p][more_b], axis=0)[:2] > [0.8, 0.8])
    assert all(
        np.mean(sim.data[utils_th_p][more_b], axis=0)[2:] < [0.01, 0.01])
    assert similarity(sim.data[utils_th_p][all_e], np.ones((1, 4))) < 0.05
    assert similarity(sim.data[utils_th_p][inhib], np.ones((1, 4))) < 0.05
    assert similarity(sim.data[out_p][more_a], vocab.parse("A").v) > 0.8
    assert similarity(sim.data[out_p][more_a], vocab.parse("B").v) > 0.8
    assert similarity(sim.data[out_p][more_b], vocab.parse("A").v) > 0.8
    assert similarity(sim.data[out_p][more_b], vocab.parse("B").v) > 0.8
    assert similarity(sim.data[out_p][all_e], vocab.parse("F").v) > 0.8
    assert similarity(sim.data[out_p][inhib], np.ones((1, D))) < 0.05


def test_am_spa_interaction(Simulator, seed, rng):
    """Make sure associative memory interacts with other SPA modules."""
    D = 16
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    D2 = int(D / 2)
    vocab2 = Vocabulary(D2, rng=rng)
    vocab2.parse('A+B+C+D')

    def input_func(t):
        return '0.49*A' if t < 0.5 else '0.79*A'

    with nengo.spa.SPA(seed=seed) as m:
        m.buf = nengo.spa.Buffer(D)
        m.input = nengo.spa.Input(buf=input_func)

        m.am = AssociativeMemory(vocab, vocab2, threshold=0.5)

        cortical_actions = nengo.spa.Actions('am = buf')
        m.c_act = nengo.spa.Cortical(cortical_actions)

    # Check to see if model builds properly. No functionality test needed
    Simulator(m)
