__copyright__ = "2013, Nengo contributors"
__license__ = "http://www.gnu.org/licenses/gpl.html"

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())  # Prevent output if no handler set

from .model import Model
from .objects import LIF
