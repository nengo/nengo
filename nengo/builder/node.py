import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.operator import Reset, SimPyFunc
from nengo.node import Node
from nengo.utils.compat import is_array_like


@Builder.register(Node)
def build_node(model, node):
    # input signal
    if not is_array_like(node.output) and node.size_in > 0:
        sig_in = Signal(np.zeros(node.size_in), name="%s.in" % node)
        model.add_op(Reset(sig_in))
    else:
        sig_in = None

    # Provide output
    if node.output is None:
        sig_out = sig_in
    elif callable(node.output):
        sig_out = (Signal(np.zeros(node.size_out), name="%s.out" % node)
                   if node.size_out > 0 else None)
        model.add_op(SimPyFunc(
            output=sig_out, fn=node.output, t_in=True, x=sig_in))
    elif is_array_like(node.output):
        sig_out = Signal(node.output, name="%s.out" % node)
    else:
        raise ValueError("Invalid node output type '%s'" % (
            node.output.__class__.__name__))

    model.sig[node]['in'] = sig_in
    model.sig[node]['out'] = sig_out
    model.params[node] = None
