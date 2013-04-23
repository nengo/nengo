#from .lif import LIFNeuron
from .lif_rate import LIFRateNeuron

types = {
#    'lif': LIFNeuron,
    'lif-rate': LIFRateNeuron,
}

__all__ = types.values()
