import numpy as np

from nengo.builder import Builder, Signal
from nengo.builder.operator import Reset
from nengo.connection import Connection, LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import BuildError
from nengo.node import Node
from nengo.probe import Probe
from nengo.utils.compat import iteritems


def conn_probe(model, probe):
    # Connection probes create a connection from the target, and probe
    # the resulting signal (used when you want to probe the default
    # output of an object, which may not have a predefined signal)

    conn = Connection(probe.target, probe, synapse=probe.synapse,
                      solver=probe.solver, add_to_container=False)

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeded[conn] = model.seeded[probe]
    model.seeds[conn] = model.seeds[probe]

    # Make a sink signal for the connection
    model.sig[probe]['in'] = Signal(np.zeros(conn.size_out), name=str(probe))
    model.add_op(Reset(model.sig[probe]['in']))

    # Build the connection
    model.build(conn)


def signal_probe(model, key, probe):
    # Signal probes directly probe a target signal

    try:
        sig = model.sig[probe.obj][key]
    except IndexError:
        raise BuildError(
            "Attribute %r is not probeable on %s." % (key, probe.obj))

    if probe.slice is not None:
        sig = sig[probe.slice]

    if probe.synapse is None:
        model.sig[probe]['in'] = sig
    else:
        model.sig[probe]['in'] = model.build(probe.synapse, sig)


probemap = {
    Ensemble: {'decoded_output': None,
               'input': 'in'},
    Neurons: {'output': None,
              'spikes': None,
              'rates': None,
              'input': 'in'},
    Node: {'output': None},
    Connection: {'output': 'weighted',
                 'input': 'in'},
    LearningRule: {},  # make LR signals probeable, but no mapping required
}


@Builder.register(Probe)
def build_probe(model, probe):
    """Builds a `.Probe` object into a model.

    Under the hood, there are two types of probes:
    connection probes and signal probes.

    Connection probes are those that are built by creating a new `.Connection`
    object from the probe's target to the probe, and calling that connection's
    build function. Creating and building a connection ensure that the result
    of probing the target's attribute is the same as would result from that
    target being connected to another object.

    Signal probes are those that are built by finding the correct `.Signal`
    in the model and calling the build function corresponding to the probe's
    synapse.

    Parameters
    ----------
    model : Model
        The model to build into.
    probe : Probe
        The connection to build.

    Notes
    -----
    Sets ``model.params[probe]`` to a list.
    `.Simulator` appends to that list when running a simulation.
    """

    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in iteritems(probemap):
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise BuildError(
            "Type %r is not probeable" % type(probe.obj).__name__)

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(model, probe)
    else:
        signal_probe(model, key, probe)

    model.probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []
