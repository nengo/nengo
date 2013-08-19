import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())  # Prevent output if no handler set

from .model import Model
from .objects import LIF
