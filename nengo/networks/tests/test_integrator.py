import nengo
from nengo.npext import rms
from nengo.processes import Piecewise


def test_integrator(Simulator, plt, seed):
    model = nengo.Network(seed=seed)
    with model:
        inputs = {0: 0, 0.2: 1, 1: 0, 2: -2, 3: 0, 4: 1, 5: 0}
        input = nengo.Node(Piecewise(inputs))

        tau = 0.1
        T = nengo.networks.Integrator(tau, n_neurons=100, dimensions=1)
        nengo.Connection(input, T.input, synapse=tau)

        A = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(A, A, synapse=tau)
        nengo.Connection(input, A, transform=tau, synapse=tau)

        input_p = nengo.Probe(input, sample_every=0.01)
        A_p = nengo.Probe(A, synapse=0.01, sample_every=0.01)
        T_p = nengo.Probe(T.ensemble, synapse=0.01, sample_every=0.01)

    with Simulator(model) as sim:
        sim.run(6.0)

    t = sim.trange(sample_every=0.01)
    plt.plot(t, sim.data[A_p], label="Manual")
    plt.plot(t, sim.data[T_p], label="Template")
    plt.plot(t, sim.data[input_p], "k", label="Input")
    plt.legend(loc="best")

    assert rms(sim.data[A_p] - sim.data[T_p]) < 0.1
