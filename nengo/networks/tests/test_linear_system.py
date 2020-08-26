import numpy as np
import pytest

import nengo
from nengo.exceptions import ValidationError
from nengo.linear_system import LinearSystem
from nengo.networks.linear_system import LinearSystemNetwork, ss2sim
from nengo.utils.numpy import nrmse


class LinearSystemNodeNetwork(LinearSystemNetwork):
    def make_state_object(self, n_neurons, dimensions, **kwargs):
        if "radius" in kwargs:
            kwargs.pop("radius")
        state = nengo.Node(size_in=dimensions, **kwargs)
        return state, state, state


class LinearSystemSReluNetwork(LinearSystemNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, neuron_type=nengo.SpikingRectifiedLinear(), **kwargs)


def _test_linear_system_network(
    sys,
    synapse,
    NetworkType,
    Simulator,
    allclose,
    seed=0,
    plt=None,
    dt=None,
    simtime=1.0,
    input_f=None,
    **kwargs
):
    probe_synapse = nengo.Alpha(0.005)

    with nengo.Network(seed=seed) as net:
        ref = nengo.Node(sys, label="ref")
        ref_p = nengo.Probe(ref, synapse=probe_synapse)

        subnet = NetworkType(sys, synapse, dt=dt, **kwargs)
        input_p = (
            nengo.Probe(subnet.input, synapse=probe_synapse) if subnet.input else None
        )
        state_p = nengo.Probe(subnet.state_output, synapse=probe_synapse)
        output_p = nengo.Probe(subnet.output, synapse=probe_synapse)

        if input_f is not None:
            inp = nengo.Node(input_f, label="inp")
            nengo.Connection(inp, ref, synapse=None)
            nengo.Connection(inp, subnet.input, synapse=None)

        # This is a "trick" for getting the output without needing to filter it.
        # It is equivalent to taking subnet.state_input as x since it is already
        # filtered. The main drawback of this approach is if input is spiking and D != 0
        # then it won't get filtered. Also, even when D is 0, it's worth noting that
        # this is creating a connection from input to output that goes through a filter
        # but doesn't get represented using neurons -- although this is just a linear
        # transformation that could be combined with weights elsewhere.
        # [ x <- filt(Ax + Bu) and y <- Cx + Du ] => [ y <- filt(CAx + CBu) + Du ]
        # This also works for getting the state without filtering it.
        # Note the output synapse needs to be initialized to Cx0 in this case, to mimic
        # what we would get from computing C.dot(filt(Ax))) with filt initialized to x0.
        output_nonfilt = nengo.Node(size_in=subnet.system.output_size)
        A, B, C, D = subnet.mapped_system.ss
        nengo.Connection(
            subnet.state_output,
            output_nonfilt,
            transform=C.dot(A),
            synapse=subnet.synapse.copy(initial_output=C.dot(subnet.mapped_system.x0)),
        )
        if subnet.input is not None:
            nengo.Connection(
                subnet.input, output_nonfilt, transform=C.dot(B), synapse=subnet.synapse
            )
            if "D" in subnet.ss_connections:
                nengo.Connection(
                    subnet.input, output_nonfilt, transform=D, synapse=None
                )
        output_nonfilt_p = nengo.Probe(output_nonfilt, synapse=None)

    with Simulator(net, dt=0.001 if dt is None else dt, progress_bar=False) as sim:
        sim.run(simtime)

    t = sim.trange()
    ref = sim.data[ref_p]
    u = sim.data[input_p] if input_p else None
    x = sim.data[state_p]
    y = sim.data[output_p]
    y_nonfilt = sim.data[output_nonfilt_p]

    if plt is not None:
        plt.plot(t, ref, "-", label="ref")
        if u is not None:
            plt.plot(t, u, ":", label="u")
        plt.plot(t, x, "-.", label="x")
        plt.plot(t, y, "--", label="y")
        plt.plot(t, y_nonfilt, "--", label="y (unfiltered)")
        plt.legend()

    assert allclose(probe_synapse.filt(y_nonfilt, dt=sim.dt), y, atol=0.1)

    return sim.trange(), ref, u, x, y


@pytest.mark.parametrize(
    "NetworkType, synapse, dt, tol",
    [
        (LinearSystemNodeNetwork, nengo.Lowpass(0.01), 0.001, 0.01),
        (LinearSystemSReluNetwork, nengo.Lowpass(0.01), 1.2e-3, 0.15),
        (LinearSystemNetwork, nengo.Alpha(0.01), None, 0.15),
    ],
)
def test_autonomous_oscillator(
    NetworkType, synapse, dt, tol, Simulator, seed, plt, allclose
):
    omega = 4 * np.pi
    A = [[0, -omega], [omega, 0]]
    C = np.eye(2)
    x0 = [1, 0]
    sys = LinearSystem((A, None, C, None), x0=x0)

    t, ref, u, x, y = _test_linear_system_network(
        sys,
        synapse,
        NetworkType,
        Simulator,
        allclose=allclose,
        dt=dt,
        seed=seed,
        plt=plt,
        n_neurons=200,
    )
    e = nrmse(y, ref, axis=0)
    assert (e < tol).all()


@pytest.mark.parametrize(
    "NetworkType, synapse, dt, tol",
    [
        (LinearSystemNodeNetwork, nengo.Lowpass(0.01), 0.001, 0.01),
        (LinearSystemSReluNetwork, nengo.Lowpass(0.01), 1.2e-3, 0.15),
        (LinearSystemNetwork, nengo.Alpha(0.01), None, 0.15),
    ],
)
def test_attractor_oscillator(
    NetworkType, synapse, dt, tol, Simulator, seed, plt, allclose
):
    inp_freqs = np.array([7.0, -15.0])
    omega = 4 * np.pi
    gain = -2.5
    A = [[gain, -omega], [omega, gain]]
    B = 3 * np.array([[1, 1], [-1, 1]])
    C = [[0, 1], [1, 0]]
    D = 0.3 * np.array([[0, -1], [1, 0]])
    x0 = [0.5, 0]
    sys = LinearSystem((A, B, C, D), x0=x0)

    t, ref, u, x, y = _test_linear_system_network(
        sys,
        synapse,
        NetworkType,
        Simulator,
        allclose=allclose,
        dt=dt,
        input_f=lambda t: np.sin(inp_freqs * t),
        seed=seed,
        plt=plt,
        n_neurons=200,
    )
    e = nrmse(y, ref, axis=0)
    assert (e < tol).all()


def test_ss2sim_errors():
    sys = LinearSystem(([1], [0.3, 1]))
    lowpass_digital = nengo.Lowpass(0.1).discretize(dt=0.001)

    with pytest.raises(ValueError, match=r"`sys\.analog` must equal `synapse\.analog`"):
        ss2sim(sys, lowpass_digital)

    highpass = nengo.synapses.Highpass(0.1)
    with pytest.raises(ValueError, match=r"analog synapse .* must not have zeros"):
        ss2sim(sys, highpass)


def test_linearsystem_errors():
    system = LinearSystem(([1], [0.3, 1]))
    synapse = nengo.Lowpass(0.1, initial_output=1)

    with pytest.raises(ValidationError, match="initial value of the synapse is unused"):
        LinearSystemNetwork(system, synapse, add_to_container=False)
