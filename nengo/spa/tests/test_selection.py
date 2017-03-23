import numpy as np
from numpy.testing import assert_allclose

import nengo
from nengo.spa.selection import ThresholdingArray, WTA


def test_thresholding_array(Simulator, plt, seed):
    with nengo.Network(seed=seed) as m:
        thresholding = ThresholdingArray(50, 4, 0.2, function=lambda x: x > 0.)
        in_node = nengo.Node([0.8, 0.5, 0.2, -0.1], label='input')
        nengo.Connection(in_node, thresholding.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(thresholding.output, synapse=0.03)
        thresholded_p = nengo.Probe(thresholding.thresholded, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()

    plt.subplot(3, 1, 1)
    plt.plot(t, sim.data[in_p])
    plt.ylabel("Input")
    plt.subplot(3, 1, 2)
    plt.plot(t, sim.data[out_p])
    plt.plot(t[t > 0.15], 0.9 * np.ones(t.shape)[t > 0.15], c='k', lw=2)
    plt.ylabel("Output")
    plt.subplot(3, 1, 3)
    plt.plot(t, sim.data[thresholded_p])
    plt.plot(t[t > 0.15], 0.6 * np.ones(t.shape)[t > 0.15], c='k', lw=2)
    plt.plot(t[t > 0.15], 0.3 * np.ones(t.shape)[t > 0.15], c='k', lw=2)
    plt.ylabel("Utilities")

    assert np.all(sim.data[out_p][t > 0.15, :2] > 0.9)
    assert np.all(sim.data[out_p][t > 0.15, 2:] < 0.001)
    assert_allclose(sim.data[thresholded_p][t > 0.15, 0], 0.6, atol=0.05)
    assert_allclose(sim.data[thresholded_p][t > 0.15, 1], 0.3, atol=0.05)
    assert np.all(sim.data[thresholded_p][t > 0.15, 2:] < 0.001)


def test_thresholding_array_output_shift(Simulator, plt, seed):
    with nengo.Network(seed=seed) as m:
        thresholding = ThresholdingArray(50, 2, 0.5)
        in_node = nengo.Node([0.7, 0.4])
        nengo.Connection(in_node, thresholding.input)
        p = nengo.Probe(thresholding.output, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.2)
    t = sim.trange()

    plt.plot(t, sim.data[p])
    plt.axhline(y=0.7)
    plt.ylim(0., 1.)
    plt.xlabel("Time")
    plt.ylabel("Output")

    assert_allclose(sim.data[p][t > 0.15, 0], 0.7, atol=0.05)
    assert_allclose(sim.data[p][:, 1], 0.)


def test_wta(Simulator, plt, seed):
    def input_func(t):
        if t < 0.2:
            return np.array([1.0, 0.8, 0.0, 0.0])
        elif t < 0.3:
            return np.zeros(4)
        else:
            return np.array([0.8, 1.0, 0.0, 0.0])

    with nengo.Network(seed=seed) as m:
        wta = WTA(50, 4, threshold=0.3, function=lambda x: x > 0.)

        in_node = nengo.Node(output=input_func, label='input')
        nengo.Connection(in_node, wta.input)

        in_p = nengo.Probe(in_node)
        out_p = nengo.Probe(wta.output, synapse=0.03)
        thresholded_p = nengo.Probe(wta.thresholded, synapse=0.03)

    with Simulator(m) as sim:
        sim.run(0.5)
    t = sim.trange()
    more_a = (t > 0.15) & (t < 0.2)
    more_b = t > 0.45

    plt.subplot(3, 1, 1)
    plt.plot(t, sim.data[in_p])
    plt.ylabel("Input")
    plt.ylim(0., 1.1)
    plt.subplot(3, 1, 2)
    plt.plot(t, sim.data[out_p])
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.9, c='g', lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.9, c='g', lw=2)
    plt.ylabel("Output")
    plt.ylim(0., 1.1)
    plt.subplot(3, 1, 3)
    plt.plot(t, sim.data[thresholded_p])
    plt.plot(t[more_a], np.ones(t.shape)[more_a] * 0.6, c='g', lw=2)
    plt.plot(t[more_b], np.ones(t.shape)[more_b] * 0.6, c='g', lw=2)
    plt.ylabel("Thresholded")
    plt.ylim(0., 1.1)

    assert np.all(sim.data[out_p][more_a, 0] > 0.9)
    assert np.all(sim.data[out_p][more_a, 1] < 0.1)
    assert np.all(sim.data[out_p][more_b, 1] > 0.9)
    assert np.all(sim.data[out_p][more_b, 0] < 0.1)
    assert np.all(sim.data[thresholded_p][more_a, 0] > 0.6)
    assert np.all(sim.data[thresholded_p][more_a, 1:] < 0.001)
    assert np.all(sim.data[thresholded_p][more_b, 1] > 0.6)
    assert np.all(sim.data[thresholded_p][more_b, 0] < 0.001)
    assert np.all(sim.data[thresholded_p][more_b, 2:] < 0.001)
