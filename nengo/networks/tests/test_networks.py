import pytest

import nengo
from nengo.exceptions import ObsoleteError


def test_net_arg_obsoleteerror():
    n_neurons = 10
    dimensions = 1
    with nengo.Network() as net:
        with pytest.raises(ObsoleteError):
            nengo.networks.CircularConvolution(n_neurons, dimensions, net=net)
        with pytest.raises(ObsoleteError):
            nengo.networks.Integrator(0.1, n_neurons, dimensions, net=net)
        with pytest.raises(ObsoleteError):
            nengo.networks.Oscillator(0.1, 1, n_neurons, net=net)
        with pytest.raises(ObsoleteError):
            nengo.networks.Product(n_neurons, dimensions, net=net)
        with pytest.raises(ObsoleteError):
            nengo.networks.InputGatedMemory(n_neurons, dimensions, net=net)

        with pytest.raises(ObsoleteError):
            nengo.networks.BasalGanglia(dimensions, net=net)
        with pytest.raises(ObsoleteError):
            nengo.networks.Thalamus(dimensions, net=net)
