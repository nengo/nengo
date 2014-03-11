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

from .nonlinearities import PythonFunction, LIF, LIFRate, Direct
from .objects import NengoObject, Ensemble, Node, Connection, Probe, Network
from . import networks
from .simulator import Simulator


logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass


def log(debug=False, path=None):
    """Log messages.

    If path is None, logging messages will be printed to the console (stdout).
    If it not None, logging messages will be appended to the file at that path.

    Typically someone using Nengo as a library will set up their own
    logging things, and Nengo will just populate their log.
    However, if the user is using Nengo directly, they can use this
    function to get log output.
    """
    level = logging.DEBUG if debug else logging.WARNING
    if logging.root.getEffectiveLevel() > level:
        logging.root.setLevel(level)

    if path is None:
        logger.info("Logging to console")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    else:
        logger.info("Logging to %s", path)
        handler = logging.FileHandler(path, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            ('%(asctime)s [%(levelname)s] %(name)s.%(funcName)s'
             '@ L%(lineno)d\n  %(message)s')))
    handler.setLevel(level)
    logging.root.addHandler(handler)


# TODO: Don't use a global variable for this
context = collections.deque(maxlen=100)  # stack of nengo.Network objects
