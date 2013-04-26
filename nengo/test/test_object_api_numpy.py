
from math import sin
from nengo.object_api import (
        LIFNeurons,
        Network,
        Probe,
        simulation_time,
        Simulator,
        TimeNode,
        )
import test_object_api

class Smoke(test_object_api.ObjectAPISmokeTests):
    def Simulator(self, *args, **kwargs):
        return Simulator(backend='numpy', *args, **kwargs)


def foo_smoke_3():
    # Learning!
    net = Network()
    net.add(Probe(simulation_time))

    tn = net.add(TimeNode(sin, name='sin'))
    ens1 = net.add(LIFNeurons(13))
    latent1 = net.add(LinearNeurons(1))
    mismatch = mse_mismatch(latent1.output_current, tn.output)
    conn = net.add(hPES_Connection(ens1.output, latent1.input_current,
                                  error_signal=mismatch.error_signal))

    net.add(Probe(mismatch.output))

    sim = Simulator(net, dt=0.001, verbosity=0, backend='numpy')
    results = sim.run(.1)

    assert len(results[simulation_time]) == 101
    total_n_spikes = 0
    for i, t in enumerate(results[simulation_time]):
        output_i = results[ens.spikes][i]
        assert len(output_i) == 13
        assert all(oi in (0, 1) for oi in output_i)
        total_n_spikes += sum(output_i)
    assert total_n_spikes > 0
