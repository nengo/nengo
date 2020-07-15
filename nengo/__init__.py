# pylint: disable=wrong-import-order,wrong-import-position

"""Nengo provides a package for doing large-scale brain modelling in Python.

The source code repository for this package is found at
https://www.github.com/nengo/nengo. Examples of models can be found
in the `examples` directory of the source code repository.
"""

import logging
import sys

from nengo.version import version as __version__

if sys.version_info < (3, 5):
    raise ImportError(
        """
You are running Python version %s with Nengo version %s.
Nengo requires at least Python 3.5.

The fact that this version was installed on your system probably means that you
are using an older version of pip; you should consider upgrading with

 $ pip install pip setuptools --upgrade

There are two options for getting Nengo working:

- Upgrade to Python >= 3.5

- Install an older version of Nengo:

 $ pip install 'nengo<3.0'
"""
        % (sys.version, __version__)
    )
del sys

# Nengo namespace (API)
from nengo.base import Process
from nengo.config import Config
from nengo.connection import Connection
from nengo.ensemble import Ensemble
from nengo.node import Node
from nengo.neurons import (
    AdaptiveLIF,
    AdaptiveLIFRate,
    Direct,
    Izhikevich,
    LIF,
    LIFRate,
    PoissonSpiking,
    RectifiedLinear,
    RegularSpiking,
    Sigmoid,
    SpikingRectifiedLinear,
    StochasticSpiking,
    Tanh,
)
from nengo.network import Network
from nengo.learning_rules import PES, BCM, Oja, Voja
from nengo.params import Default
from nengo.probe import Probe
from nengo.rc import rc, RC_DEFAULTS
from nengo.simulator import Simulator
from nengo.synapses import Alpha, LinearFilter, Lowpass, Triangle
from nengo.transforms import Convolution, Dense, Sparse
from nengo import dists, exceptions, networks, presets, processes, utils
from nengo import spa  # pylint: disable=cyclic-import

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass

__copyright__ = "2013-2020, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
