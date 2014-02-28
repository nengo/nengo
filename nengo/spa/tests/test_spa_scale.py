from __future__ import print_function

import numpy as np
import pytest

import nengo
from nengo import spa
from nengo.utils.testing import Plotter


def test_spa_basic(Simulator):  # noqa: C901

    class SpaTestBasic(spa.SPA):
        class Rules:
            def a():
                match(state='A')  # noqa: F821
                effect(state='B', state2=state*10)  # noqa: F821

            def b():
                match(state='B')  # noqa: F821
                effect(state='C', state2=state*10)  # noqa: F821

            def c():
                match(state='C')  # noqa: F821
                effect(state='D', state2=state*10)  # noqa: F821

            def d():
                match(state='D')  # noqa: F821
                effect(state='E', state2=state*10)  # noqa: F821

            def e():
                match(state='E')  # noqa: F821
                effect(state='A', state2=state*10)  # noqa: F821

        def __init__(self):
            super(SpaTestBasic, self).__init__()
            self.state = spa.Memory(dimensions=32)
            self.state2 = spa.Memory(dimensions=32)

            self.bg = spa.BasalGanglia(rules=self.Rules)
            self.thal = spa.Thalamus(self.bg)

            def state_input(t):
                if t < 0.1:
                    return 'A'
                else:
                    return '0'
            self.input = spa.Input(state=state_input)

    model = nengo.Network()
    with model:
        s = SpaTestBasic(label='spa')
        print(s._modules)

        pState = nengo.Probe(s.state.state.output, 'output', synapse=0.03)
        pState2 = nengo.Probe(s.state2.state.output, 'output', synapse=0.03)
        pRules = nengo.Probe(s.thal.rules.output, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(1)

    vectors = s.get_module_output('state')[1].vectors.T

    with Plotter(Simulator) as plt:
        plt.subplot(3, 1, 1)
        plt.plot(np.dot(sim.data[pState], vectors))
        plt.subplot(3, 1, 2)
        plt.plot(np.dot(sim.data[pState2], vectors))
        plt.subplot(3, 1, 3)
        plt.plot(sim.data[pRules])
        plt.savefig('test_spa_scale.test_spa_basic.pdf')


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
