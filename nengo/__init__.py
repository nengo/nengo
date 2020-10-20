# pylint: disable=wrong-import-order,wrong-import-position

"""Nengo provides a package for doing large-scale brain modelling in Python.

The source code repository for this package is found at
https://www.github.com/nengo/nengo. Examples of models can be found
in the `examples` directory of the source code repository.
"""

import logging
import sys

# Nengo namespace (API)
from nengo import spa  # pylint: disable=cyclic-import
from nengo import dists, exceptions, networks, presets, processes, utils
from nengo.base import Process
from nengo.config import Config
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.learning_rules import BCM, PES, RLS, Oja, Voja
from nengo.network import Network
from nengo.neurons import (
    LIF,
    AdaptiveLIF,
    AdaptiveLIFRate,
    Direct,
    Izhikevich,
    LIFRate,
    PoissonSpiking,
    RectifiedLinear,
    RegularSpiking,
    Sigmoid,
    SpikingRectifiedLinear,
    StochasticSpiking,
    Tanh,
)
from nengo.node import Node
from nengo.params import Default
from nengo.probe import Probe
from nengo.rc import RC_DEFAULTS, rc
from nengo.simulator import Simulator
from nengo.synapses import Alpha, LinearFilter, Lowpass, Triangle
from nengo.transforms import Convolution, Dense, Sparse
from nengo.version import version as __version__

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass

__copyright__ = "2013-2020, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
