import logging
log = logging.getLogger(__name__)
try:
    log.addHandler(logging.NullHandler())  # Prevent output if no handler set
except AttributeError:
    # -- older Python's don't have NullHandler.
    pass

from .model import Model
from .objects import LIF
