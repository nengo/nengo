import numpy as np
import pytest

import nengo
from nengo.utils.optimization import NNeuronsForAccuracy


# TODO different neuron types, higher dimensional function
@pytest.mark.slow
def test_n_neurons_for_accuracy(Simulator, plt, seed, logger, rng):
    with nengo.Network(seed=seed) as model:
        ens = nengo.Ensemble(1, 1, encoders=[[1]], intercepts=[-1])
        nengo.Connection(nengo.Node(1), ens)
        p = nengo.Probe(ens)

    with Simulator(model) as sim:
        sim.run(0.5)

    noise = np.sqrt(np.mean(np.square(sim.data[p])))
    logger.info('noise = %f', noise)

    with nengo.Network(seed=seed) as model:
        def stimulus(t):
            return (t / 5.) * 2. - 1.

        fn = lambda x: x * x
        ens = nengo.Ensemble(NNeuronsForAccuracy(
            fn, .01, noise=noise, rng=rng), 1)
        output = nengo.Node(size_in=1)
        nengo.Connection(nengo.Node(stimulus), ens)
        nengo.Connection(ens, output, function=fn)
        p = nengo.Probe(output, synapse=None)

    with Simulator(model) as sim:
        sim.run(5.)

    target = fn(stimulus(sim.trange()))
    sel = sim.trange() > 0.4
    mse = np.mean(np.square(target[sel] - np.squeeze(sim.data[p][sel])))
    # shift = np.mean(target[sel] - sim.data[p][sel])
    # mse2 = np.mean(np.square(target[sel] - shift - sim.data[p][sel]))

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p])
    plt.plot(sim.trange(), target)
    plt.subplot(2, 1, 2)
    plt.plot(target[sel] - np.squeeze(sim.data[p][sel]))
    logger.info(
        '%i neurons, mse=%f', sim.model.params[ens].deferred['n_neurons'],
        mse)
    assert mse <= .01
