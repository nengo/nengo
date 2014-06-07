import numpy as np
import pytest

import nengo
from nengo.spa import Vocabulary
from nengo.spa.assoc_mem import AssociativeMemory


def test_am_defaults(Simulator):
    """Default assoc memory: auto-associative, threshold = 0.3,
       non-inhibitable, non-wta, does not output utilities or thresholded
       utilities.
    """

    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab)
        in_node = nengo.Node(output=vocab.parse("A").v,
                             label='input')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][-10:], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("A").v,
                       atol=.1, rtol=.01)


def test_am_different_output_vocab(Simulator):
    """Standard associative memory (differing input and output vocabularies),
       threshold = 0.3, non-inhibitable, non-wta, does not output utilities
       or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    D2 = int(D / 2)
    vocab2 = Vocabulary(D2, rng=rng)
    vocab2.parse('A+B+C+D')

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, vocab2)
        in_node = nengo.Node(output=vocab.parse("A").v,
                             label='input')
        out_node = nengo.Node(size_in=D2, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][-10:], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab2.parse("A").v,
                       atol=.1, rtol=.01)


def test_am_default_output(Simulator):
    """Auto-associative memory that outputs a predefined vector if no match
       is made to the vocabulary in the AM, threshold = 0.3, non-inhibitable,
       non-wta, does not output utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D+E+F')

    vocab2 = vocab.create_subset(["A", "B", "C", "D"])

    def input_func(t):
        if t < 0.5:
            return vocab.parse('A').v
        else:
            return vocab.parse('E').v

    m = nengo.Network('model', seed=123)
    with m:
        # Create AMem that outputs "D" if no match is found
        am = AssociativeMemory(vocab2,
                               default_output_vector=vocab.parse('F').v)
        in_node = nengo.Node(output=input_func, label='input')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("E").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("F").v,
                       atol=.1, rtol=.01)


def test_am_threshold(Simulator):
    """Auto-associative memory, threshold = 0.5, non-inhibitable,
       non-wta, does not output utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    def input_func(t):
        if t < 0.5:
            return vocab.parse('0.49*A').v
        else:
            return vocab.parse('0.76*A').v

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, threshold=0.5)
        in_node = nengo.Node(output=input_func, label='input')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("0.49*A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("0.76*A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab.parse("0").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("A").v,
                       atol=.1, rtol=.01)


def test_am_inhibit(Simulator):
    """Auto-associative memory, threshold = 0.3, inhibitable,
       non-wta, does not output utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    def inhib_func(t):
        return int(t > 0.5)

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, inhibitable=True)
        in_node = nengo.Node(output=vocab.parse('A').v, label='input')
        inhib_node = nengo.Node(output=inhib_func, label='inhib')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("0").v,
                       atol=.1, rtol=.01)


def test_am_wta(Simulator):
    """Auto-associative memory, threshold = 0.3, non-inhibitable,
       wta, does not output utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    def input_func(t):
        if t < 0.5:
            return vocab.parse('A+0.8*B').v
        elif t < 0.6:
            return vocab.parse("0").v
        else:
            return vocab.parse('0.8*A+B').v

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, wta_output=True)
        in_node = nengo.Node(output=input_func, label='input')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("A+0.8*B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("0.8*A+B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("B").v,
                       atol=.1, rtol=.01)


def test_am_utilities(Simulator):
    """Auto-associative memory, threshold = 0.3, non-inhibitable,
       non-wta, outputs utilities and thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    def input_func(t):
        if t < 0.5:
            return vocab.parse('A+0.8*B').v
        else:
            return vocab.parse('0.8*A+B').v

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, output_utilities=True,
                               output_thresholded_utilities=True)
        in_node = nengo.Node(output=input_func, label='input')
        utils_node = nengo.Node(size_in=4, label='utils')
        utils_th_node = nengo.Node(size_in=4, label='utils_th')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.utilities, utils_node, synapse=0.05)
        nengo.Connection(am.thresholded_utilities, utils_th_node, synapse=0.05)

        in_p = nengo.Probe(in_node)
        utils_p = nengo.Probe(utils_node)
        utils_th_p = nengo.Probe(utils_th_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("A+0.8*B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("0.8*A+B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[utils_p][490:500], [1, 0.75, 0, 0],
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[utils_p][-10:], [0.75, 1, 0, 0],
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[utils_th_p][490:500], [1.05, 1.05, 0, 0],
                       atol=.1, rtol=.05)
    assert np.allclose(sim.data[utils_th_p][-10:], [1.05, 1.05, 0, 0],
                       atol=.1, rtol=.05)


def test_am_default_output_inhibit(Simulator):
    """Auto-associative memory, defaults to predefined vector if no match is
       found, threshold = 0.3, inhibitable, non-wta, does not output utilities
       or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D+E+F')

    vocab2 = vocab.create_subset(["A", "B", "C", "D"])

    def inhib_func(t):
        return int(t > 0.5)

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab2,
                               default_output_vector=vocab.parse("F").v,
                               inhibitable=True)
        in_node = nengo.Node(output=vocab.parse('E').v, label='input')
        inhib_node = nengo.Node(output=inhib_func, label='inhib')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("E").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("E").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab.parse("F").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("0").v,
                       atol=.1, rtol=.01)


def test_am_default_output_wta(Simulator):
    """Auto-associative memory, defaults to predefined vector if no match is
       found, threshold = 0.3, non-inhibitable, wta, does not output utilities
       or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D+E+F')

    vocab2 = vocab.create_subset(["A", "B", "C", "D"])

    def input_func(t):
        if t < 0.3:
            return vocab.parse('A+0.8*B').v
        elif t < 0.6:
            return vocab.parse("E").v
        else:
            return vocab.parse('0.8*A+B').v

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab2,
                               default_output_vector=vocab.parse("F").v,
                               wta_output=True)
        in_node = nengo.Node(output=input_func, label='input')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][290:300], vocab.parse("A+0.8*B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][590:600], vocab.parse("E").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("0.8*A+B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][290:300], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][590:600], vocab.parse("F").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("B").v,
                       atol=.1, rtol=.01)


def test_am_inhib_wta(Simulator):
    """Auto-associative memory, threshold = 0.3, inhibitable,
       wta, does not output utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    def inhib_func(t):
        if t < 0.25:
            return 0
        elif t < 0.5:
            return 1
        elif t < 0.75:
            return 0
        else:
            return 1

    def input_func(t):
        if t < 0.5:
            return vocab.parse('A+0.8*B').v
        else:
            return vocab.parse('0.8*A+B').v

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, inhibitable=True, wta_output=True)
        in_node = nengo.Node(output=input_func, label='input')
        inhib_node = nengo.Node(output=inhib_func, label='inhib')
        out_node = nengo.Node(size_in=D, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("A+0.8*B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("0.8*A+B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][240:250], vocab.parse("A").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab.parse("0").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][740:750], vocab.parse("B").v,
                       atol=.1, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab.parse("0").v,
                       atol=.1, rtol=.01)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
