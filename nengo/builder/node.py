import numpy as np

from nengo.builder import Builder, Signal
from nengo.builder.operator import Reset, SimPyFunc
from nengo.exceptions import BuildError
from nengo.node import Node
from nengo.processes import Process
from nengo.utils.compat import is_array_like


@Builder.register(Node)
def build_node(model, node):
    """Builds a `.Node` object into a model.

    The node build function is relatively simple. It involves creating input
    and output signals, and connecting them with an `.Operator` that depends
    on the type of ``node.output``.

    Parameters
    ----------
    model : Model
        The model to build into.
    node : Node
        The node to build.

    Notes
    -----
    Sets ``model.params[node]`` to ``None``.
    """

    # input signal
    if not is_array_like(node.output) and node.size_in > 0:
        sig_in = Signal(np.zeros(node.size_in), name="%s.in" % node)
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    # Provide output
    if node.output is None:
        sig_out = sig_in
    elif isinstance(node.output, Process):
        sig_out = Signal(np.zeros(node.size_out), name="%s.out" % node)
        model.build(node.output, sig_in, sig_out)
    elif callable(node.output):
        sig_out = (Signal(np.zeros(node.size_out), name="%s.out" % node)
                   if node.size_out > 0 else None)
        model.add_op(SimPyFunc(
            output=sig_out, fn=node.output, t=model.time, x=sig_in))
    elif is_array_like(node.output):
        sig_out = Signal(node.output, name="%s.out" % node)
    else:
        raise BuildError(
            "Invalid node output type %r" % node.output.__class__.__name__)

    model.sig[node]['in'] = sig_in
    model.sig[node]['out'] = sig_out
    model.params[node] = None
