from .lif import LIFNeuron
from .lif_rate import LIFRateNeuron
from .neuron import accumulate

types = {
    'lif': LIFNeuron,
    'lif-rate': LIFRateNeuron,
}

__all__ = types.values() + [accumulate]
