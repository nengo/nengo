import nengo
from nengo.spa import Vocabulary
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
