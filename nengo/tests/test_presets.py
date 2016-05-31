import numpy as np

import nengo


def test_thresholding_preset(Simulator, seed, plt):
    threshold = 0.3
    with nengo.Network(seed) as model:
        with nengo.presets.ThresholdingEnsembles(threshold):
            ens = nengo.Ensemble(50, 1)
        stimulus = nengo.Node(lambda t: t)
        nengo.Connection(stimulus, ens)
        p = nengo.Probe(ens, synapse=0.01)

    with Simulator(model) as sim:
        sim.run(1.)

    plt.plot(sim.trange(), sim.trange(), label="optimal")
    plt.plot(sim.trange(), sim.data[p], label="actual")
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.title("Threshold = {}".format(threshold))
    plt.legend(loc='best')

    se = np.square(np.squeeze(sim.data[p]) - sim.trange())

    assert np.allclose(sim.data[p][sim.trange() < threshold], 0.0)
    assert np.sqrt(np.mean(se[sim.trange() > 0.5])) < 0.05
