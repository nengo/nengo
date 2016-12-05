import pytest

import nengo
from nengo.builder.neurons import SimNeurons


@pytest.mark.parametrize(
    "SpikingType",
    (nengo.RegularSpiking, nengo.PoissonSpiking, nengo.StochasticSpiking),
)
def test_spiking_builders(SpikingType):
    # use a base type with its own state(s), to make sure those states get built
    neuron_type = SpikingType(nengo.AdaptiveLIFRate())

    with nengo.Network() as net:
        neurons = nengo.Ensemble(10, 1, neuron_type=neuron_type).neurons

    with nengo.Simulator(net) as sim:
        ops = [op for op in sim.model.operators if isinstance(op, SimNeurons)]
        assert len(ops) == 1

        adaptation = sim.model.sig[neurons]["adaptation"]
        # All signals get put in `sets`
        assert sum(adaptation is sig for sig in ops[0].sets) == 1
