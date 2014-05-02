"""
Nengo
=====

Nengo provides a package for doing large-scale brain modelling in Python.

The source code repository for this package is found at
https://www.github.com/ctn-waterloo/nengo. Examples of models can be found
in the `examples` directory of the source code repository.
"""

__copyright__ = "2013-2014, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
from .version import version as __version__

import logging

# Nengo namespace (API)
from .config import Config, Default
from .objects import Ensemble, Node, Connection, Probe, Network
from .neurons import Direct, LIF, LIFRate, AdaptiveLIF, AdaptiveLIFRate
from .learning_rules import PES, BCM, Oja
from . import networks
from .simulator import Simulator
from .utils.logging import log

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass
