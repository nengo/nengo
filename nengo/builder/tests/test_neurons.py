import numpy as np
import pytest

import nengo
from nengo.builder.neurons import SimNeurons
from nengo.builder.tests.test_operator import _test_operator_arg_attributes


@pytest.mark.parametrize(
    "SpikingType",
    (nengo.RegularSpiking, nengo.PoissonSpiking, nengo.StochasticSpiking),
)
def test_spiking_builders(SpikingType):
    # use a base type with its own state(s), to make sure those states get built
    neuron_type = SpikingType(
        nengo.AdaptiveLIFRate(initial_state={"adaptation": np.ones(10) * 2}),
        initial_state={"voltage": np.ones(10) * 3}
        if SpikingType is nengo.RegularSpiking
        else {},
    )

    with nengo.Network() as net:
        neurons = nengo.Ensemble(10, 1, neuron_type=neuron_type).neurons

        # check that the expected attributes are probeable
        nengo.Probe(neurons, "output")
        nengo.Probe(neurons, "rate_out")
        nengo.Probe(neurons, "adaptation")
        if SpikingType is nengo.RegularSpiking:
            nengo.Probe(neurons, "voltage")

    with nengo.Simulator(net) as sim:
        ops = [op for op in sim.step_order if isinstance(op, SimNeurons)]
        assert len(ops) == 2
        assert ops[0].neurons is neurons.ensemble.neuron_type.base_type
        assert ops[1].neurons is neurons.ensemble.neuron_type

        adaptation = sim.model.sig[neurons]["adaptation"]
        # All signals get put in `sets`
        assert sum(adaptation is sig for sig in ops[0].sets) == 1

        # check that initial state argument is applied correctly
        assert np.allclose(sim.signals[sim.model.sig[neurons]["adaptation"]], 2)
        if SpikingType is nengo.RegularSpiking:
            assert np.allclose(sim.signals[sim.model.sig[neurons]["voltage"]], 3)


def test_simneurons():
    argnames = ["neurons", "J", "output"]
    non_signals = ["neurons"]
    _, sim = _test_operator_arg_attributes(
        SimNeurons, argnames, non_signals=non_signals
    )

    assert str(sim) == "SimNeurons{neurons, J, output}"
