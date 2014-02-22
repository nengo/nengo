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
import copy
import logging
import sys
import threading
import weakref

from .model import Model
from .nonlinearities import PythonFunction, LIF, LIFRate, Direct
from .objects import Ensemble, Node, Connection, Probe
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


def monkeypatch_deepcopy():
    """Monkey-patch for deepcopying weakrefs.

    Python 2.6 and below had an issue in which you couldn't deepcopy weakrefs.
    This monkeypatch fixes the issue for Python 2.6.
    Note that this only patches weakref.ref, since that's all we use.
    This patch, and the rest of the weakref patches if necessary,
    were taken from coopr.pyomo:
      https://projects.coin-or.org/Coopr/browser/coopr.pyomo/
        trunk/coopr/pyomo/base/PyomoModel.py
    """
    assert sys.version_info[0] == 2 and sys.version_info[1] <= 6
    copy._copy_dispatch[weakref.ref] = copy._copy_immutable
    copy._deepcopy_dispatch[weakref.ref] = copy._deepcopy_atomic

if sys.version_info[0] == 2 and sys.version_info[1] <= 6:
    monkeypatch_deepcopy()

context = threading.local()
context.model = Model('default')
