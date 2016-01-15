import nengo
from nengo.utils.functions import piecewise
from nengo.utils.numpy import rmse


def test_oscillator(Simulator, plt, seed):
    model = nengo.Network(seed=seed)
    with model:
        inputs = {0: [1, 0], 0.5: [0, 0]}
        input = nengo.Node(piecewise(inputs), label='Input')

        tau = 0.1
        freq = 5
        T = nengo.networks.Oscillator(tau, freq,  n_neurons=100)
        nengo.Connection(input, T.input)

        A = nengo.Ensemble(100, dimensions=2)
        nengo.Connection(A, A, synapse=tau,
                         transform=[[1, -freq*tau], [freq*tau, 1]])
        nengo.Connection(input, A)

        in_probe = nengo.Probe(input, sample_every=0.01)
        A_probe = nengo.Probe(A, synapse=0.01, sample_every=0.01)
        T_probe = nengo.Probe(T.ensemble, synapse=0.01, sample_every=0.01)

    with Simulator(model) as sim:
        sim.run(3.0)

    t = sim.trange(dt=0.01)
    plt.plot(t, sim.data[A_probe], label='Manual')
    plt.plot(t, sim.data[T_probe], label='Template')
    plt.plot(t, sim.data[in_probe], 'k', label='Input')
    plt.legend(loc='best')

    assert rmse(sim.data[A_probe], sim.data[T_probe]) < 0.2
