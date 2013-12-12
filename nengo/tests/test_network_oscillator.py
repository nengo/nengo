import logging

import pytest

import nengo
import nengo.helpers
from nengo.tests.helpers import Plotter, rmse


logger = logging.getLogger(__name__)


def test_oscillator(Simulator, nl):
    model = nengo.Model('Oscillator')

    inputs = {0: [1, 0], 0.5: [0, 0]}
    input = nengo.Node(nengo.helpers.piecewise(inputs), label='Input')

    tau = 0.1
    freq = 5
    T = nengo.networks.Oscillator(
        tau, freq, label="Oscillator", neurons=nl(100))
    nengo.Connection(input, T.input)

    A = nengo.Ensemble(nl(100), label='A', dimensions=2)
    nengo.Connection(A, A, filter=tau,
                     transform=[[1, -freq*tau], [freq*tau, 1]])
    nengo.Connection(input, A)

    in_probe = nengo.Probe(input, "output")
    A_probe = nengo.Probe(A, "decoded_output", filter=0.01)
    T_probe = nengo.Probe(T.ensemble, "decoded_output", filter=0.01)

    sim = Simulator(model)
    sim.run(3.0)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange()
        plt.plot(t, sim.data(A_probe), label='Manual')
        plt.plot(t, sim.data(T_probe), label='Template')
        plt.plot(t, sim.data(in_probe), 'k', label='Input')
        plt.legend(loc=0)
        plt.savefig('test_oscillator.test_oscillator.pdf')
        plt.close()

    assert rmse(sim.data(A_probe), sim.data(T_probe)) < 0.3


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
