from .network import Network

basic = [Network]

from .ensemble import Ensemble
from .input import Input
#from .simplenode import SimpleNode

from .origin import Origin
from .ensemble_origin import EnsembleOrigin

from .learned_termination import LearnedTermination

advanced = [Ensemble, Input, Origin,
            EnsembleOrigin, LearnedTermination]

__all__ = basic + advanced
