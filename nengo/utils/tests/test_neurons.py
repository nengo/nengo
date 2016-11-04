import pytest

import nengo.utils.neurons
from nengo.exceptions import MovedError


def test_neurons_moved():
    with pytest.raises(MovedError):
        nengo.utils.neurons.spikes2events()
    with pytest.raises(MovedError):
        nengo.utils.neurons._rates_isi_events()
    with pytest.raises(MovedError):
        nengo.utils.neurons.rates_isi()
    with pytest.raises(MovedError):
        nengo.utils.neurons.lowpass_filter()
    with pytest.raises(MovedError):
        nengo.utils.neurons.rates_kernel()
    with pytest.raises(MovedError):
        nengo.utils.neurons.settled_firingrate()
