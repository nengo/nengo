import numpy as np

import nengo


def test_basic(Simulator, plt, seed):
    with nengo.Network(seed=seed) as net:
        bg = nengo.networks.BasalGanglia(dimensions=5)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p = nengo.Probe(bg.output, synapse=0.01)

    with Simulator(net) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > -0.1
    assert np.all(output[1:] < -0.8)


def test_thalamus(Simulator, plt, seed):
    with nengo.Network(seed=seed) as net:
        bg = nengo.networks.BasalGanglia(dimensions=5)
        input = nengo.Node([0.8, 0.4, 0.4, 0.4, 0.4], label="input")
        nengo.Connection(input, bg.input, synapse=None)

        thal = nengo.networks.Thalamus(dimensions=5)
        nengo.Connection(bg.output, thal.input)

        p = nengo.Probe(thal.output, synapse=0.01)

    with Simulator(net) as sim:
        sim.run(0.2)

    t = sim.trange()
    output = np.mean(sim.data[p][t > 0.1], axis=0)

    plt.plot(t, sim.data[p])
    plt.ylabel("Output")

    assert output[0] > 0.8
    assert np.all(output[1:] < 0.01)


def test_bias_input(Simulator, plt, seed, allclose):
    with nengo.Network(seed=seed) as net:
        bg = nengo.networks.BasalGanglia(dimensions=3, input_bias=-0.5)
        input = nengo.Node([0.8, 0.5, 0.2], label="input")
        nengo.Connection(input, bg.input, synapse=None)
        p_in = nengo.Probe(bg.input, synapse=0.01)

    with Simulator(net) as sim:
        sim.run(0.1)

    t = sim.trange()
    plt.plot(t, sim.data[p_in])
    plt.ylabel("Input")

    assert allclose(sim.data[p_in][t > 0.08], [0.3, 0, -0.3], atol=0.005)


def test_overridden_configs():
    # Using default synapses, all that are not None should be
    # Lowpass with tau of 0.002 (AMPA) or 0.008 (GABA)
    bg = nengo.networks.BasalGanglia(dimensions=2)
    for connection in bg.all_connections:
        if connection.synapse is None:
            continue
        assert isinstance(connection.synapse, nengo.Lowpass)
        assert connection.synapse.tau in (0.002, 0.008)

    # These configs do not override the synapse on the connection.
    # The synapses should be the same as before, and the config should
    # be unchanged after instantiation, even though inside the class
    # it is changed.
    ampa_config = nengo.Config(nengo.Connection)
    ampa_config[nengo.Connection].solver = nengo.solvers.LstsqL2nz()
    assert "synapse" not in ampa_config[nengo.Connection]
    gaba_config = nengo.Config(nengo.Connection)
    gaba_config[nengo.Connection].solver = nengo.solvers.LstsqL2nz()
    assert "synapse" not in gaba_config[nengo.Connection]

    bg = nengo.networks.BasalGanglia(
        dimensions=2, ampa_config=ampa_config, gaba_config=gaba_config)
    for connection in bg.all_connections:
        if connection.synapse is None:
            continue
        # Same synapses
        assert isinstance(connection.synapse, nengo.Lowpass)
        assert connection.synapse.tau in (0.002, 0.008)
        # But solver should be different
        assert isinstance(connection.solver, nengo.solvers.LstsqL2nz)
    # Ensure default synapse has been removed from both objects
    assert "synapse" not in ampa_config[nengo.Connection]
    assert "synapse" not in gaba_config[nengo.Connection]

    # This config overrides the synapse on AMPA and GABA connections
    config_with_synapse = nengo.Config(nengo.Connection)
    config_with_synapse[nengo.Connection].synapse = nengo.Alpha(0.1)
    bg = nengo.networks.BasalGanglia(
        dimensions=2,
        ampa_config=config_with_synapse,
        gaba_config=config_with_synapse,
    )
    for connection in bg.all_connections:
        if connection.synapse is None:
            continue
        # Synapse changes
        assert isinstance(connection.synapse, nengo.Alpha)
        assert connection.synapse.tau == 0.1
        # Solver should be the same
        assert not isinstance(connection.solver, nengo.solvers.LstsqL1)
    # Ensure synapse is the same as before passing to BasalGanglia
    synapse = config_with_synapse[nengo.Connection].synapse
    assert isinstance(synapse, nengo.Alpha)
    assert synapse.tau == 0.1
