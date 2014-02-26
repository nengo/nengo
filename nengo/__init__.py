"""
Nengo
=====

Nengo provides a package for doing large-scale brain modelling in Python.

The source code repository for this package is found at
https://www.github.com/ctn-waterloo/nengo. Examples of models can be found
in the `examples` directory of the source code repository.
"""

__title__ = "nengo"
__author__ = "CNRGlab UWaterloo"
__license__ = "http://www.gnu.org/licenses/gpl.html"
__copyright__ = "Copyrigt 2013, Nengo Contributors"
from .version import version as __version__

import collections
import logging
import sys

from .api import PythonFunction, LIF, LIFRate, Direct, Ensemble, Node, \
    Connection, Probe, Model, Gaussian, Uniform
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
