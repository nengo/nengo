"""These are helper functions that various backends may find useful for
generating their own Builder system.
"""
from __future__ import absolute_import
import numpy as np

import nengo


def generate_graphviz(objs, connections):
    """Create a .gv file with this set of objects and connections

    Parameters
    ----------
    objs : list of Nodes and Ensembles
        All the objects in the model
    connections : list of Connections
        All the Connections in the model

    Returns the text contents of the desired .dot file

    This can be useful for debugging and testing Builders that manipulate
    the model graph before construction.
    """
    text = []
    text.append('digraph G {')
    for obj in objs:
        text.append('  "%d" [label="%s"];' % (id(obj), obj.label))

    def label(transform):
        # determine the label for a connection based on its transform
        transform = np.asarray(transform)
        if len(transform.shape) == 0:
            return ''
        return '%dx%d' % transform.shape

    for c in connections:
        text.append('  "%d" -> "%d" [label="%s"];' % (
            id(c.pre), id(c.post), label(c.transform)))
    text.append('}')
    return '\n'.join(text)


def remove_passthrough_nodes(objs, connections):  # noqa: C901
    """Returns a version of the model without passthrough Nodes

    For some backends (such as SpiNNaker), it is useful to remove Nodes that
    have 'None' as their output.  These nodes simply sum their inputs and
    use that as their output. These nodes are defined purely for organizational
    purposes and should not affect the behaviour of the model.  For example,
    the 'input' and 'output' Nodes in an EnsembleArray, which are just meant to
    aggregate data.

    Note that removing passthrough nodes can simplify a model and may be useful
    for other backends as well.  For example, an EnsembleArray connected to
    another EnsembleArray with an identity matrix as the transform
    should collapse down to D Connections between the corresponding Ensembles
    inside the EnsembleArrays.

    Parameters
    ----------
    objs : list of Nodes and Ensembles
        All the objects in the model
    connections : list of Connections
        All the Connections in the model

    Returns the objs and connections of the resulting model.  The passthrough
    Nodes will be removed, and the Connections that interact with those Nodes
    will be replaced with equivalent Connections that don't interact with those
    Nodes.

    """

    inputs, outputs = find_all_io(objs, connections)
    result_conn = list(connections)
    result_objs = list(objs)

    # look for passthrough Nodes to remove
    for obj in objs:
        if isinstance(obj, nengo.Node) and obj.output is None:
            result_objs.remove(obj)

            # get rid of the connections to and from this Node
            for c in inputs[obj]:
                result_conn.remove(c)
                outputs[c.pre].remove(c)
            for c in outputs[obj]:
                result_conn.remove(c)
                inputs[c.post].remove(c)

            # replace those connections with equivalent ones
            for c_in in inputs[obj]:
                if c_in.pre is obj:
                    raise Exception('Cannot remove a Node with feedback')

                for c_out in outputs[obj]:
                    c = _create_replacement_connection(c_in, c_out)
                    if c is not None:
                        result_conn.append(c)
                        outputs[c.pre].append(c)  # put this in the list, since
                        inputs[c.post].append(c)  # it might be used another
                                                  # time through the loop

    return result_objs, result_conn


def find_all_io(objs, connections):
    """Build up a list of all inputs and outputs for each object"""
    inputs = {}
    outputs = {}
    for obj in objs:
        inputs[obj] = []
        outputs[obj] = []
    for c in connections:
        inputs[c.post].append(c)
        outputs[c.pre].append(c)
    return inputs, outputs


def _create_replacement_connection(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post is c_out.pre
    assert c_in.post.output is None

    # determine the filter for the new Connection
    if c_in.filter is None:
        filter = c_out.filter
    elif c_out.filter is None:
        filter = c_in.filter
    else:
        raise NotImplementedError('Cannot merge two filters')
        # Note: the algorithm below is in the right ballpark,
        #  but isn't exactly the same as two low-pass filters
        #filter = c_out.filter + c_in.filter

    function = c_in.function
    if c_out.function is not None:
        raise Exception('Cannot remove a Node with a' +
                        'function being computed on it')

    # compute the combined transform
    transform = np.dot(c_out.transform, c_in.transform)
    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.all(transform == 0):
        return None

    dummy = nengo.Model()  # need a dummy model so these
    with dummy:            # connections don't get added
        args = {}
        if function is not None:
            args['function'] = function
        c = nengo.Connection(c_in.pre, c_out.post,
                             filter=filter,
                             transform=transform, **args)
    return c
