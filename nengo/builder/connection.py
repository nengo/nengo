import collections

import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.node import build_pyfunc
from nengo.builder.operator import DotInc, ProdUpdate, Reset
from nengo.builder.signal import Signal
from nengo.connection import Connection
from nengo.ensemble import Ensemble, Neurons
from nengo.neurons import Direct
from nengo.node import Node
from nengo.utils.builder import full_transform
import nengo.utils.numpy as npext


BuiltConnection = collections.namedtuple(
    'BuiltConnection', ['decoders', 'eval_points', 'transform', 'solver_info'])


def build_linear_system(model, conn, rng):
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = conn.eval_points
    if eval_points is None:
        eval_points = npext.array(
            model.params[conn.pre_obj].eval_points, min_dims=2)
    else:
        eval_points = npext.array(eval_points, min_dims=2)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    activities = model.dt * conn.pre_obj.neuron_type.rates(x, gain, bias)
    if np.count_nonzero(activities) == 0:
        raise RuntimeError(
            "In '%s', for '%s', 'activites' matrix is all zero. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (str(conn), str(conn.pre_obj)))

    if conn.function is None:
        # TODO: slice eval_points here rather than later, as it could
        # significantly reduce computation
        targets = eval_points
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            targets[i] = conn.function(ep)

    return eval_points, activities, targets


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    if isinstance(conn.pre_obj, Neurons):
        model.sig[conn]['in'] = model.sig[conn.pre_obj.ensemble]["neuron_out"]
    else:
        model.sig[conn]['in'] = model.sig[conn.pre_obj]["out"]

    if isinstance(conn.post_obj, Neurons):
        model.sig[conn]['out'] = model.sig[conn.post_obj.ensemble]["neuron_in"]
    else:
        model.sig[conn]['out'] = model.sig[conn.post_obj]["in"]

    decoders = None
    eval_points = None
    solver_info = None
    transform = full_transform(conn)

    # Figure out the signal going across this connection
    if (isinstance(conn.pre_obj, Node) or
            (isinstance(conn.pre_obj, Ensemble) and
             isinstance(conn.pre_obj.neuron_type, Direct))):
        # Node or Decoded connection in directmode
        if conn.function is None:
            signal = model.sig[conn]['in']
        else:
            sig_in, signal = build_pyfunc(
                fn=lambda x: conn.function(x[conn.pre_slice]),
                t_in=False,
                n_in=model.sig[conn]['in'].size,
                n_out=conn.size_out,
                label=conn.label,
                model=model)
            model.add_op(DotInc(model.sig[conn]['in'],
                                model.sig['common'][1],
                                sig_in,
                                tag="%s input" % conn.label))
    elif isinstance(conn.pre_obj, Ensemble):
        # Normal decoded connection
        eval_points, activities, targets = build_linear_system(
            model, conn, rng=rng)

        if conn.solver.weights:
            # account for transform
            targets = np.dot(targets, transform.T)
            transform = np.array(1., dtype=np.float64)

            decoders, solver_info = conn.solver(
                activities, targets, rng=rng,
                E=model.params[conn.post_obj].scaled_encoders.T)
            model.sig[conn]['out'] = model.sig[conn.post_obj]['neuron_in']
            signal_size = model.sig[conn]['out'].size
        else:
            decoders, solver_info = conn.solver(activities, targets, rng=rng)
            signal_size = conn.size_mid

        # Add operator for decoders
        decoders = decoders.T

        model.sig[conn]['decoders'] = Signal(
            decoders, name="%s.decoders" % conn.label)
        signal = Signal(np.zeros(signal_size), name=conn.label)
        model.add_op(ProdUpdate(model.sig[conn]['decoders'],
                                model.sig[conn]['in'],
                                model.sig['common'][0],
                                signal,
                                tag="%s decoding" % conn.label))
    else:
        # Direct connection
        signal = model.sig[conn]['in']

    # Add operator for filtering
    if conn.synapse is not None:
        model.sig[conn]['synapse_in'] = signal
        model.build(conn, conn.synapse, 'synapse')
        signal = model.sig[conn]['synapse_out']

    if conn.modulatory:
        # Make a new signal, effectively detaching from post
        model.sig[conn]['out'] = Signal(
            np.zeros(model.sig[conn]['out'].size),
            name="%s.mod_output" % conn.label)
        model.add_op(Reset(model.sig[conn]['out']))

    # Add operator for transform
    if isinstance(conn.post_obj, Neurons):
        if not model.has_built(conn.post_obj.ensemble):
            # Since it hasn't been built, it wasn't added to the Network,
            # which is most likely because the Neurons weren't associated
            # with an Ensemble.
            raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                               "that are not a part of any Ensemble." % (
                                   conn, conn.post_obj))
        transform *= model.params[conn.post_obj.ensemble].gain[:, np.newaxis]

    model.sig[conn]['transform'] = Signal(transform,
                                          name="%s.transform" % conn.label)
    model.add_op(DotInc(model.sig[conn]['transform'],
                        signal,
                        model.sig[conn]['out'],
                        tag=conn.label))

    if conn.learning_rule:
        # Forcing update of signal that is modified by learning rules.
        # Learning rules themselves apply DotIncs.

        if isinstance(conn.pre_obj, Neurons):
            modified_signal = model.sig[conn]['transform']
        elif isinstance(conn.pre_obj, Ensemble):
            if conn.solver.weights:
                # TODO: make less hacky.
                # Have to do this because when a weight_solver
                # is provided, then learning rules should operators on
                # "decoders" which is really the weight matrix.
                model.sig[conn]['transform'] = model.sig[conn]['decoders']
                modified_signal = model.sig[conn]['transform']
            else:
                modified_signal = model.sig[conn]['decoders']
        else:
            raise TypeError("Can't apply learning rules to connections of "
                            "this type. pre type: %s, post type: %s"
                            % (type(conn.pre_obj).__name__,
                               type(conn.post_obj).__name__))

        model.add_op(ProdUpdate(model.sig['common'][0],
                                model.sig['common'][0],
                                model.sig['common'][1],
                                modified_signal,
                                tag="Learning Rule Dummy Update"))

    model.params[conn] = BuiltConnection(decoders=decoders,
                                         eval_points=eval_points,
                                         transform=transform,
                                         solver_info=solver_info)
