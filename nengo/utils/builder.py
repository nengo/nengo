"""These are helper functions that various backends may find useful for
generating their own Builder system.
"""

from __future__ import absolute_import
import collections

import numpy as np

import nengo


def full_transform(conn, allow_scalars=True):
    """Compute the full transform for a connection.

    Parameters
    ----------
    conn : Connection
        The connection for which to compute the full transform.
    allow_scalars : boolean, optional
        If true (default), will not make scalars into full transforms when
        not using slicing, since these work fine in the reference builder.
        If false, these scalars will be turned into scaled identity matrices.
    """
    transform = conn.transform

    if conn._preslice == slice(None) and conn._postslice == slice(None):
        if transform.ndim == 2:
            # transform is already full, so return a copy
            return np.array(transform)
        elif transform.size == 1 and allow_scalars:
            return np.array(transform)

    full_size_in = (conn._pre.size_out if conn.function is None
                    else conn.size_in)
    full_size_out = conn._post.size_in

    # Create the new transform matching the pre/post dimensions
    new_transform = np.zeros((full_size_out, full_size_in))
    if transform.ndim < 2:
        slice_to_list = lambda s, d: (
            np.arange(d)[s] if isinstance(s, slice) else s)
        preslice = slice_to_list(conn._preslice, full_size_in)
        postslice = slice_to_list(conn._postslice, full_size_out)
        new_transform[postslice, preslice] = transform
    else:  # if transform.ndim == 2:
        repeated_inds = lambda x: (
            not isinstance(x, slice) and np.unique(x).size != len(x))
        if repeated_inds(conn._preslice):
            raise ValueError("Input object selection has repeated indices")
        if repeated_inds(conn._postslice):
            raise ValueError("Output object selection has repeated indices")
        rows_transform = np.array(new_transform[conn._postslice])
        rows_transform[:, conn._preslice] = transform
        new_transform[conn._postslice] = rows_transform
        # Note: the above is a little obscure, but we do it so that lists of
        #  indices can specify selections of rows and columns, rather than
        #  just individual items

    return new_transform


def objs_and_connections(network):
    """Given a Network, returns all (ensembles + nodes, connections)."""
    objs = list(network.ensembles + network.nodes)
    connections = list(network.connections)
    for subnetwork in network.networks:
        subobjs, subconnections = objs_and_connections(subnetwork)
        objs.extend(subobjs)
        connections.extend(subconnections)
    return objs, connections


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

    inputs, outputs = find_all_io(connections)
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
                        # put this in the list, since it might be used
                        # another time through the loop
                        outputs[c.pre].append(c)
                        inputs[c.post].append(c)

    return result_objs, result_conn


def find_all_io(connections):
    """Build up a list of all inputs and outputs for each object"""
    inputs = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    for c in connections:
        inputs[c.post].append(c)
        outputs[c.pre].append(c)
    return inputs, outputs


def _create_replacement_connection(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post is c_out.pre
    assert c_in.post.output is None

    # determine the filter for the new Connection
    if c_in.synapse is None:
        synapse = c_out.synapse
    elif c_out.synapse is None:
        synapse = c_in.synapse
    else:
        raise NotImplementedError('Cannot merge two filters')
        # Note: the algorithm below is in the right ballpark,
        #  but isn't exactly the same as two low-pass filters
        # filter = c_out.filter + c_in.filter

    function = c_in.function
    if c_out.function is not None:
        raise Exception('Cannot remove a Node with a '
                        'function being computed on it')

    # compute the combined transform
    transform = np.dot(full_transform(c_out), full_transform(c_in))
    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.all(transform == 0):
        return None

    c = nengo.Connection(c_in.pre, c_out.post,
                         synapse=synapse,
                         transform=transform,
                         function=function,
                         add_to_container=False)
    return c
