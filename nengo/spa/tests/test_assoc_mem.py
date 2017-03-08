import nengo
import numpy as np
from nengo import spa
from nengo.spa import Vocabulary
from nengo.spa.assoc_mem import ThresholdingAssocMem, WtaAssocMem
from nengo.spa.utils import similarity


filtered_step_fn = lambda x: np.maximum(1. - np.exp(-15. * x), 0.)


def test_am_basic(Simulator, plt, seed, rng):
    """Basic associative memory test."""

    d = 64
    vocab = Vocabulary(d, rng=rng)
    vocab.populate('A; B; C; D')

    with spa.Module('model', seed=seed) as m:
        m.am = ThresholdingAssocMem(threshold=0.3, input_vocab=vocab,
                                    function=filtered_step_fn)
        m.stimulus = spa.Input()
        m.stimulus.am = 'A'

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()

    plt.subplot(3, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(3, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[t > 0.15], np.ones(t.shape)[t > 0.15] * 0.95, c='g', lw=2)
    plt.ylabel("Output")

    assert np.all(similarity(sim.data[in_p][t > 0.15], vocab)[:, 0] > 0.99)
    assert np.all(similarity(sim.data[out_p][t > 0.15], vocab)[:, 0] > 0.95)
    assert np.all(similarity(sim.data[out_p][t > 0.15], vocab)[:, 1:] < 0.1)


def test_am_threshold(Simulator, plt, seed, rng):
    """Associative memory thresholding with differing input/output vocabs."""
    d = 64
    vocab = Vocabulary(d, rng=rng)
    vocab.populate('A; B; C; D')

    d2 = int(d / 2)
    vocab2 = Vocabulary(d2, rng=rng)
    vocab2.populate('A; B; C; D')

    def input_func(t):
        return '0.49 * A' if t < 0.1 else '0.8 * B'

    with spa.Module('model', seed=seed) as m:
        m.am = ThresholdingAssocMem(
            threshold=0.5, input_vocab=vocab, output_vocab=vocab2,
            function=filtered_step_fn)
        m.stimulus = spa.Input()
        m.stimulus.am = input_func

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.3)
    t = sim.trange()
    below_th = t < 0.1
    above_th = t > 0.25

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab2))
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.9, c='g', lw=2)
    plt.ylabel("Output")

    assert np.mean(sim.data[out_p][below_th]) < 0.01
    assert np.all(
        similarity(sim.data[out_p][above_th], [vocab2['B'].v]) > 0.90)


def test_am_wta(Simulator, plt, seed, rng):
    """Test the winner-take-all ability of the associative memory."""

    d = 64
    vocab = Vocabulary(d, rng=rng)
    vocab.populate('A; B; C; D')

    def input_func(t):
        if t < 0.2:
            return 'A + 0.8 * B'
        elif t < 0.3:
            return '0'
        else:
            return '0.8 * A + B'

    with spa.Module('model', seed=seed) as m:
        m.am = WtaAssocMem(
            threshold=0.3, input_vocab=vocab, function=filtered_step_fn)
        m.stimulus = spa.Input()
        m.stimulus.am = input_func

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()
    more_a = (t > 0.15) & (t < 0.2)
    more_b = t > 0.45

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.ylim(top=1.1)
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.9, c='g', lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.9, c='g', lw=2)
    plt.ylabel("Output")

    assert np.all(similarity(sim.data[out_p][more_a], [vocab['A'].v]) > 0.9)
    assert np.all(similarity(sim.data[out_p][more_a], [vocab['B'].v]) < 0.1)
    assert np.all(similarity(sim.data[out_p][more_b], [vocab['B'].v]) > 0.79)
    assert np.all(similarity(sim.data[out_p][more_b], [vocab['A'].v]) < 0.1)


def test_am_default_output(Simulator, plt, seed, rng):
    d = 64
    vocab = Vocabulary(d, rng=rng)
    vocab.populate('A; B; C; D')

    def input_func(t):
        return '0.2 * A' if t < 0.25 else 'A'

    with spa.Module('model', seed=seed) as m:
        m.am = ThresholdingAssocMem(threshold=0.5, input_vocab=vocab,
                                    function=filtered_step_fn)
        m.am.add_default_output('D', 0.5)
        m.stimulus = spa.Input()
        m.stimulus.am = input_func

        in_p = nengo.Probe(m.am.input)
        out_p = nengo.Probe(m.am.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()
    below_th = (t > 0.15) & (t < 0.25)
    above_th = t > 0.4

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab))
    plt.ylabel("Input")
    plt.subplot(2, 1, 2)
    plt.plot(t, similarity(sim.data[out_p], vocab))
    plt.plot(t[below_th], np.ones(t.shape)[below_th] * 0.9, c='c', lw=2)
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.9, c='b', lw=2)
    plt.plot(t[above_th], np.ones(t.shape)[above_th] * 0.1, c='c', lw=2)
    plt.ylabel("Output")

    assert np.all(similarity(sim.data[out_p][below_th], [vocab['D'].v]) > 0.9)
    assert np.all(similarity(sim.data[out_p][above_th], [vocab['D'].v]) < 0.1)
    assert np.all(similarity(sim.data[out_p][above_th], [vocab['A'].v]) > 0.9)


def test_am_spa_keys_as_expressions(Simulator, plt, seed, rng):
    """Provide semantic pointer expressions as input and output keys."""
    d = 64

    vocab_in = Vocabulary(d, rng=rng)
    vocab_out = Vocabulary(d, rng=rng)

    vocab_in.populate('A; B')
    vocab_out.populate('C; D')

    in_keys = ['A', 'A*B']
    out_keys = ['C*D', 'C+D']

    with nengo.spa.Module(seed=seed) as model:
        model.am = ThresholdingAssocMem(
            threshold=0.3, input_vocab=vocab_in, output_vocab=vocab_out,
            input_keys=in_keys, output_keys=out_keys)

        model.inp = spa.Input()
        model.inp.am = lambda t: 'A' if t < 0.1 else 'A*B'

        in_p = nengo.Probe(model.am.input)
        out_p = nengo.Probe(model.am.output, synapse=0.03)

    with nengo.Simulator(model) as sim:
        sim.run(0.2)

    # Specify t ranges
    t = sim.trange()
    t_item1 = (t > 0.075) & (t < 0.1)
    t_item2 = (t > 0.175) & (t < 0.2)

    # Modify vocabularies (for plotting purposes)
    vocab_in.add(in_keys[1], vocab_in.parse(in_keys[1]).v)
    vocab_out.add(out_keys[0], vocab_out.parse(out_keys[0]).v)

    plt.subplot(2, 1, 1)
    plt.plot(t, similarity(sim.data[in_p], vocab_in))
    plt.ylabel("Input: " + ', '.join(in_keys))
    plt.legend(vocab_in.keys, loc='best')
    plt.ylim(top=1.1)
    plt.subplot(2, 1, 2)
    for t_item, c, k in zip([t_item1, t_item2], ['b', 'g'], out_keys):
        plt.plot(t, similarity(
            sim.data[out_p], [vocab_out.parse(k).v], normalize=True),
            label=k, c=c)
        plt.plot(t[t_item], np.ones(t.shape)[t_item] * 0.9, c=c, lw=2)
    plt.ylabel("Output: " + ', '.join(out_keys))
    plt.legend(loc='best')

    assert np.mean(similarity(sim.data[out_p][t_item1],
                              vocab_out.parse(out_keys[0]).v,
                              normalize=True)) > 0.9
    assert np.mean(similarity(sim.data[out_p][t_item2],
                              vocab_out.parse(out_keys[1]).v,
                              normalize=True)) > 0.9
