import numpy as np
import pytest

import nengo
from nengo.spa import Vocabulary
from nengo.spa.assoc_mem import AssociativeMemory


def test_am_defaults(Simulator):
    """Default assoc memory.

    Options: auto-associative, threshold = 0.3, non-inhibitable, non-wta,
    does not output utilities or thresholded utilities.
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


def test_am_assoc_mem_threshold(Simulator):
    """Standard associative memory (differing input and output vocabularies).

    Options: threshold = 0.5, non-inhibitable, non-wta, does not output
    utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    D2 = int(D / 2)
    vocab2 = Vocabulary(D2, rng=rng)
    vocab2.parse('A+B+C+D')

    def input_func(t):
        if t < 0.5:
            return vocab.parse('0.49*A').v
        else:
            return vocab.parse('0.79*A').v

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab, vocab2, threshold=0.5)
        in_node = nengo.Node(output=input_func, label='input')
        out_node = nengo.Node(size_in=D2, label='output')
        nengo.Connection(in_node, am.input)
        nengo.Connection(am.output, out_node, synapse=0.03)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)

    sim = Simulator(m)
    sim.run(1.0)

    assert np.allclose(sim.data[in_p][490:500], vocab.parse("0.49*A").v,
                       atol=.15, rtol=.01)
    assert np.allclose(sim.data[in_p][-10:], vocab.parse("0.79*A").v,
                       atol=.15, rtol=.01)
    assert np.allclose(sim.data[out_p][490:500], vocab2.parse("0").v,
                       atol=.15, rtol=.01)
    assert np.allclose(sim.data[out_p][-10:], vocab2.parse("A").v,
                       atol=.15, rtol=.01)


def test_am_default_output_inhibit_utilities(Simulator):
    """Auto-associative memory (non-wta) complex test.

    Options: defaults to predefined vector if no match is found,
    threshold = 0.3, inhibitable, non-wta, outputs utilities and thresholded
    utilities.
    """
    rng = np.random.RandomState(1)

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

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab2,
                               default_output_vector=vocab.parse("F").v,
                               inhibitable=True, output_utilities=True,
                               output_thresholded_utilities=True)
        in_node = nengo.Node(output=input_func, label='input')
        inhib_node = nengo.Node(output=inhib_func, label='inhib')
        out_node = nengo.Node(size_in=D, label='output')
        utils_node = nengo.Node(size_in=4, label='utils')
        utils_th_node = nengo.Node(size_in=4, label='utils_th')
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)
        nengo.Connection(am.output, out_node, synapse=0.03)
        nengo.Connection(am.utilities, utils_node, synapse=0.05)
        nengo.Connection(am.thresholded_utilities, utils_th_node, synapse=0.05)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)
        utils_p = nengo.Probe(utils_node)
        utils_th_p = nengo.Probe(utils_th_node)

    sim = Simulator(m)
    sim.run(1.0)

    t = sim.trange()
    t1 = (t >= 0.2) & (t < 0.25)
    t2 = (t >= 0.45) & (t < 0.5)
    t3 = (t >= 0.7) & (t < 0.75)
    t4 = (t >= 0.95)
    assert np.allclose(sim.data[in_p][t1], vocab.parse("A+0.8*B").v, atol=0.1)
    assert np.allclose(sim.data[in_p][t2], vocab.parse("0.8*A+B").v, atol=0.1)
    assert np.allclose(sim.data[in_p][t3], vocab.parse("E").v, atol=0.1)
    assert np.allclose(sim.data[in_p][t4], vocab.parse("E").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t1], vocab.parse("A+B").v, atol=0.11)
    assert np.allclose(sim.data[out_p][t2], vocab.parse("A+B").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t3], vocab.parse("F").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t4], vocab.parse("0").v, atol=0.1)
    assert np.allclose(sim.data[utils_p][t1], [1, 0.75, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_p][t2], [0.75, 1, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_p][t3], [0, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_p][t4], [0, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_th_p][t1], [1.05, 1.05, 0, 0], atol=0.2)
    assert np.allclose(sim.data[utils_th_p][t2], [1.05, 1.05, 0, 0], atol=0.15)
    assert np.allclose(sim.data[utils_th_p][t3], [0, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_th_p][t4], [0, 0, 0, 0], atol=0.1)


def test_am_default_output_inhibit_utilities_wta(Simulator):
    """Auto-associative memory (wta) complex test.

    Options: defaults to predefined vector if no match is found,
    threshold = 0.3, inhibitable, wta, outputs utilities and thresholded
    utilities.
    """
    rng = np.random.RandomState(1)

    D = 64
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D+E+F')

    vocab2 = vocab.create_subset(["A", "B", "C", "D"])

    def input_func(t):
        if t < 0.25:
            return vocab.parse('A+0.8*B').v
        elif t < 0.5:
            return vocab.parse('E').v
        else:
            return vocab.parse('0.8*A+B').v

    def inhib_func(t):
        if t < 0.75:
            return 0
        else:
            return 1

    m = nengo.Network('model', seed=123)
    with m:
        am = AssociativeMemory(vocab2, wta_output=True,
                               default_output_vector=vocab.parse("F").v,
                               inhibitable=True, output_utilities=True,
                               output_thresholded_utilities=True)
        in_node = nengo.Node(output=input_func, label='input')
        inhib_node = nengo.Node(output=inhib_func, label='inhib')
        out_node = nengo.Node(size_in=D, label='output')
        utils_node = nengo.Node(size_in=4, label='utils')
        utils_th_node = nengo.Node(size_in=4, label='utils_th')
        nengo.Connection(in_node, am.input)
        nengo.Connection(inhib_node, am.inhibit)
        nengo.Connection(am.output, out_node, synapse=0.03)
        nengo.Connection(am.utilities, utils_node, synapse=0.05)
        nengo.Connection(am.thresholded_utilities, utils_th_node, synapse=0.05)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(out_node)
        utils_p = nengo.Probe(utils_node)
        utils_th_p = nengo.Probe(utils_th_node)

    sim = Simulator(m)
    sim.run(1.0)

    t = sim.trange()
    t1 = (t >= 0.2) & (t < 0.25)
    t2 = (t >= 0.45) & (t < 0.5)
    t3 = (t >= 0.7) & (t < 0.75)
    t4 = (t >= 0.95)
    assert np.allclose(sim.data[in_p][t1], vocab.parse("A+0.8*B").v, atol=0.1)
    assert np.allclose(sim.data[in_p][t2], vocab.parse("E").v, atol=0.1)
    assert np.allclose(sim.data[in_p][t3], vocab.parse("0.8*A+B").v, atol=0.1)
    assert np.allclose(sim.data[in_p][t4], vocab.parse("0.8*A+B").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t1], vocab.parse("A").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t2], vocab.parse("F").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t3], vocab.parse("B").v, atol=0.1)
    assert np.allclose(sim.data[out_p][t4], vocab.parse("0").v, atol=0.1)
    assert np.allclose(sim.data[utils_p][t1], [1, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_p][t2], [0, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_p][t3], [0, 1, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_p][t4], [0, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_th_p][t1], [1.05, 0, 0, 0], atol=0.15)
    assert np.allclose(sim.data[utils_th_p][t2], [0, 0, 0, 0], atol=0.1)
    assert np.allclose(sim.data[utils_th_p][t3], [0, 1.05, 0, 0], atol=0.15)
    assert np.allclose(sim.data[utils_th_p][t4], [0, 0, 0, 0], atol=0.1)


def test_am_spa_interaction(Simulator):
    """Standard associative memory interacting with other SPA modules.

    Options: threshold = 0.5, non-inhibitable, non-wta, does not output
    utilities or thresholded utilities.
    """
    rng = np.random.RandomState(1)

    D = 16
    vocab = Vocabulary(D, rng=rng)
    vocab.parse('A+B+C+D')

    D2 = int(D / 2)
    vocab2 = Vocabulary(D2, rng=rng)
    vocab2.parse('A+B+C+D')

    def input_func(t):
        if t < 0.5:
            return '0.49*A'
        else:
            return '0.79*A'

    m = nengo.spa.SPA('model', seed=123)
    with m:
        m.buf = nengo.spa.Buffer(D)
        m.input = nengo.spa.Input(buf=input_func)

        m.am = AssociativeMemory(vocab, vocab2, threshold=0.5)

        cortical_actions = nengo.spa.Actions('am = buf')
        m.c_act = nengo.spa.Cortical(cortical_actions)

    # Check to see if model builds properly. No functionality test needed
    Simulator(m)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
