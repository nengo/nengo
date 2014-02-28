import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.utils.testing import Plotter


def test_spa_complex(Simulator):  # noqa: C901
    model = nengo.Network()

    dimensions = 64

    class ParseWrite(spa.SPA):
        class Rules:
            def verb():
                match(vision='WRITE')  # noqa: F821
                effect(verb=vision)  # noqa: F821

            def noun():
                match(vision='ONE+TWO+THREE')  # noqa: F821
                effect(noun=vision)  # noqa: F821

            def write():
                match(vision='0.5*(NONE-WRITE-ONE-TWO-THREE)',  # noqa: F821
                      phrase='0.5*WRITE*VERB')
                effect(motor=phrase*'~NOUN')  # noqa: F821

        class CorticalRules:
            def noun():
                effect(phrase=noun*'NOUN')  # noqa: F821

            def verb():
                effect(phrase=verb*'VERB')  # noqa: F821

        def __init__(self):
            super(ParseWrite, self).__init__()
            self.vision = spa.Buffer(dimensions=dimensions)
            self.phrase = spa.Buffer(dimensions=dimensions)
            self.motor = spa.Buffer(dimensions=dimensions)

            self.noun = spa.Memory(dimensions=dimensions)
            self.verb = spa.Memory(dimensions=dimensions)

            self.bg = spa.BasalGanglia(rules=self.Rules)
            self.thal = spa.Thalamus(self.bg)

            def input_vision(t):
                sequence = ('WRITE ONE NONE WRITE TWO NONE THREE WRITE '
                            'NONE'.split())
                index = int(t / 0.5) % len(sequence)
                return sequence[index]
            self.input = spa.Input(vision=input_vision)

            self.cortical = spa.Cortical(self.CorticalRules)

    with model:
        s = ParseWrite(label='SPA')

        probes = {
            'vision': nengo.Probe(s.vision.state.output, synapse=0.03),
            'phrase': nengo.Probe(s.phrase.state.output, synapse=0.03),
            'motor': nengo.Probe(s.motor.state.output, synapse=0.03),
            'noun': nengo.Probe(s.noun.state.output, synapse=0.03),
            'verb': nengo.Probe(s.verb.state.output, synapse=0.03),
        }
    sim = Simulator(model)
    sim.run(4.5)

    with Plotter(Simulator) as plt:
        for i, module in enumerate('vision noun verb phrase motor'.split()):
            plt.subplot(5, 1, i+1)
            plt.plot(np.dot(sim.data[probes[module]],
                            s.get_module_output(module)[1].vectors.T))
            plt.legend(s.get_module_output(module)[1].keys,
                       fontsize='xx-small')
            plt.ylabel(module)
        plt.savefig('test_spa_complex.pdf')
        plt.close()


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
