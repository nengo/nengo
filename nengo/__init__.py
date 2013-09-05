import logging
import sys

from .model import Model
from .core import LIF


logger = logging.getLogger(__name__)
try:
    logger.addHandler(logging.NullHandler())  # Prevent output if no handler set
except AttributeError:
    pass


def log(debug=False):
    """Log messages to the console (stdout).

    Typically someone using Nengo as a library will set up their own
    logging things, and Nengo will just populate their log.
    However, if the user is using Nengo directly, they can use this
    functions to get a simple log output to the console.
    """
    sh = logging.StreamHandler(sys.stdout)
    level = logging.DEBUG if debug else logging.WARNING
    if logging.root.getEffectiveLevel() > level:
        logging.root.setLevel(level)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.root.addHandler(sh)
    logger.info("Logging to console")


def log_to_file(fname, debug=False):
    """Log messages to a file.

    Typically someone using Nengo as a library will set up their own
    logging things, and Nengo will just populate their log.
    However, if the user is using Nengo directly, they can use this
    functions to get a simple log output to a file.
    """
    fh = logging.FileHandler(fname, encoding='utf-8')
    level = logging.DEBUG if debug else logging.WARNING
    if logging.root.getEffectiveLevel() > level:
        logging.root.setLevel(level)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        ('%(asctime)s [%(levelname)s] %(name)s.%(funcName)s @ L%(lineno)d\n'
         '  %(message)s')))
    logging.root.addHandler(fh)
    logger.info("Logging to %s", fname)
