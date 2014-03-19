"""
Nengo
=====

Nengo provides a package for doing large-scale brain modelling in Python.

The source code repository for this package is found at
https://www.github.com/ctn-waterloo/nengo. Examples of models can be found
in the `examples` directory of the source code repository.
"""

__copyright__ = "2013, Nengo contributors"
__license__ = "http://www.gnu.org/licenses/gpl.html"
from .version import version as __version__

import collections
import logging
import sys

# Nengo namespace (API)
from .model import Model
from .neurons import LIF, LIFRate, Direct
from .objects import Ensemble, Node, Connection, Probe, Network
from . import networks
from .simulator import Simulator
from .utils.logging import log

logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass


class ContextStack(collections.deque):
    def add_to_current(self, obj):
        try:
            curr = self.__getitem__(-1)
        except IndexError:
            raise IndexError("Context has not been set")

        if not hasattr(curr, "add"):
            raise AttributeError("Current context has no add function")

        curr.add(obj)

context = ContextStack(maxlen=100)
