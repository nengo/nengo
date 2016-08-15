import nengo
import numpy as np
from nengo.spa import Vocabulary, Input
from nengo.spa.utils import similarity
from nengo.spa.assoc_mem import AssociativeMemory


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

    with nengo.spa.Module(seed=seed) as m:
        m.buf = nengo.spa.Buffer(D, vocab=vocab)
        m.input = nengo.spa.Input(buf=input_func)

        m.am = AssociativeMemory(vocab, vocab2,
                                 input_keys=['A', 'B', 'C'],
                                 output_keys=['B', 'C', 'D'],
                                 default_output_key='A',
                                 threshold=0.5,
                                 inhibitable=True,
                                 wta_output=True,
                                 threshold_output=True)

        cortical_actions = nengo.spa.Actions('am = buf')
        m.c_act = nengo.spa.Cortical(cortical_actions)

    # Check to see if model builds properly. No functionality test needed
    with Simulator(m):
        pass


def test_am_spa_keys_as_expressions(Simulator, plt, seed, rng):
    """Provide semantic pointer expressions as input and output keys."""
    D = 64

    vocab_in = Vocabulary(D, rng=rng)
    vocab_out = Vocabulary(D, rng=rng)

    vocab_in.parse('A+B')
    vocab_out.parse('C+D')

    in_keys = ['A', 'A*B']
    out_keys = ['C*D', 'C+D']

    with nengo.spa.SPA(seed=seed) as model:
        model.am = AssociativeMemory(input_vocab=vocab_in,
                                     output_vocab=vocab_out,
                                     input_keys=in_keys,
                                     output_keys=out_keys)

        model.inp = Input(am=lambda t: 'A' if t < 0.1 else 'A*B')

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
    plt.plot(t, similarity(sim.data[out_p], vocab_out))
    plt.plot(t[t_item1], np.ones(t.shape)[t_item1] * 0.9, c='r', lw=2)
    plt.plot(t[t_item2], np.ones(t.shape)[t_item2] * 0.91, c='g', lw=2)
    plt.plot(t[t_item2], np.ones(t.shape)[t_item2] * 0.89, c='b', lw=2)
    plt.ylabel("Output: " + ', '.join(out_keys))
    plt.legend(vocab_out.keys, loc='best')

    assert np.mean(similarity(sim.data[out_p][t_item1],
                              vocab_out.parse(out_keys[0]).v,
                              normalize=True)) > 0.9
    assert np.mean(similarity(sim.data[out_p][t_item2],
                              vocab_out.parse(out_keys[1]).v,
                              normalize=True)) > 0.9
