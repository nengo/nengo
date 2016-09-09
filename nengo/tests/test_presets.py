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


def test_thresholding_preset_radius(Simulator, seed):
    threshold = 0.3
    radius = 0.5
    with nengo.Network(seed) as model:
        with nengo.presets.ThresholdingEnsembles(threshold, radius=radius):
            ens = nengo.Ensemble(50, 1)

    with Simulator(model) as sim:
        pass

    assert ens.radius == radius
    assert np.all(threshold <= sim.data[ens].eval_points)
    assert np.all(sim.data[ens].eval_points <= radius)
    assert np.all(threshold <= sim.data[ens].intercepts)
    assert np.all(sim.data[ens].intercepts <= radius)
