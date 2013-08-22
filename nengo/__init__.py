import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())  # Prevent output if no handler set

from .model import Model
from .objects import LIF


try:
    import IPython.core.getipython
    ipy = IPython.core.getipython.get_ipython()
except ImportError:
    ipy = None

# -- register an automatic visualizer for IPython
if ipy:
    import simulator
    import simulator_svg
    simulator.Simulator._repr_svg_ = simulator_svg.simulator_to_svg
    print 'registering handler for simulator.Simulator'


