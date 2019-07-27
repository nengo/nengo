"""These are helper functions that various backends may find useful for
generating their own Builder system.
"""

import collections

import numpy as np

import nengo
from nengo.exceptions import MovedError, Unconvertible, ValidationError


def full_transform(conn, slice_pre=True, slice_post=True, allow_scalars=True):
    """Compute the full transform matrix for a Dense connection transform.

    Parameters
    ----------
    conn : Connection
        The connection for which to compute the full transform.
    slice_pre : boolean, optional (True)
        Whether to compute the pre slice as part of the transform.
    slice_post : boolean, optional (True)
        Whether to compute the post slice as part of the transform.
    allow_scalars : boolean, optional (True)
        If true (default), will not make scalars into full transforms when
        not using slicing, since these work fine in the reference builder.
        If false, these scalars will be turned into scaled identity matrices.
    """

    if not isinstance(conn.transform, nengo.Dense):
        raise ValidationError(
            "full_transform can only be applied to Dense " "transforms",
            attr="transform",
            obj=conn,
        )

    transform = conn.transform.init
    pre_slice = conn.pre_slice if slice_pre and conn.function is None else slice(None)
    post_slice = conn.post_slice if slice_post else slice(None)

    eq_none_slice = lambda s: isinstance(s, slice) and s == slice(None)
    if eq_none_slice(pre_slice) and eq_none_slice(post_slice):
        if transform.ndim == 2:
            # transform is already full, so return a copy
            return np.array(transform)
        elif transform.size == 1 and allow_scalars:
            if transform.ndim == 1:
                return np.array(transform[0])
            else:
                return np.array(transform)

    # Create the new transform matching the pre/post dimensions
    func_size = conn.function_info.size
    size_in = (
        (conn.pre_obj.size_out if func_size is None else func_size)
        if slice_pre
        else conn.size_mid
    )
    size_out = conn.post_obj.size_in if slice_post else conn.size_out
    new_transform = np.zeros((size_out, size_in))

    if transform.ndim < 2:
        new_transform[
            np.arange(size_out)[post_slice], np.arange(size_in)[pre_slice]
        ] = transform
        return new_transform
    elif transform.ndim == 2:
        repeated_inds = lambda x: (
            not isinstance(x, slice) and np.unique(x).size != len(x)
        )
        if repeated_inds(pre_slice):
            raise NotImplementedError("Input object selection has repeated indices")
        if repeated_inds(post_slice):
            raise NotImplementedError("Output object selection has repeated indices")

        rows_transform = np.array(new_transform[post_slice])
        rows_transform[:, pre_slice] = transform
        new_transform[post_slice] = rows_transform
        # Note: the above is a little obscure, but we do it so that lists of
        #  indices can specify selections of rows and columns, rather than
        #  just individual items
        return new_transform
    else:
        raise ValidationError(
            "Transforms with > 2 dims not supported", attr="transform", obj=conn
        )


def default_n_eval_points(n_neurons, dimensions):
    """A heuristic to determine an appropriate number of evaluation points.

    This is used by builders to generate a sufficiently large sample
    from a vector space in order to solve for accurate decoders.

    Parameters
    ----------
    n_neurons : int
        The number of neurons in the ensemble that will be sampled.
        For a connection, this would be the number of neurons in the
        `pre` ensemble.
    dimensions : int
        The number of dimensions in the ensemble that will be sampled.
        For a connection, this would be the number of dimensions in the
        `pre` ensemble.
    """
    return max(np.clip(500 * dimensions, 750, 2500), 2 * n_neurons)


def objs_and_connections(network):
    """Given a Network, returns all (ensembles + nodes, connections)."""
    return network.all_ensembles + network.all_nodes, network.all_connections


def generate_graphviz(*args, **kwargs):
    raise MovedError(location="nengo_extras.graphviz")


def _create_replacement_connection(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post_obj is c_out.pre_obj
    assert c_in.post_obj.output is None

    # determine the filter for the new Connection
    if c_in.synapse is None:
        synapse = c_out.synapse
    elif c_out.synapse is None:
        synapse = c_in.synapse
    else:
        raise Unconvertible("Cannot merge two filters")
        # Note: the algorithm below is in the right ballpark,
        #  but isn't exactly the same as two low-pass filters
        # filter = c_out.filter + c_in.filter

    function = c_in.function
    if c_out.function is not None:
        raise Unconvertible("Cannot remove a connection with a function")

    # compute the combined transform
    transform = np.dot(full_transform(c_out), full_transform(c_in))

    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.all(transform == 0):
        return None

    c = nengo.Connection(
        c_in.pre_obj,
        c_out.post_obj,
        synapse=synapse,
        transform=transform,
        function=function,
        add_to_container=False,
    )
    return c


def remove_passthrough_nodes(  # noqa: C901
    objs, connections, create_connection_fn=None
):
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
    if create_connection_fn is None:
        create_connection_fn = _create_replacement_connection

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
                outputs[c.pre_obj].remove(c)
            for c in outputs[obj]:
                result_conn.remove(c)
                inputs[c.post_obj].remove(c)

            # replace those connections with equivalent ones
            for c_in in inputs[obj]:
                if c_in.pre_obj is obj:
                    raise Unconvertible(
                        "Cannot remove a Node with a feedback connection"
                    )

                for c_out in outputs[obj]:
                    c = create_connection_fn(c_in, c_out)
                    if c is not None:
                        result_conn.append(c)
                        # put this in the list, since it might be used
                        # another time through the loop
                        outputs[c.pre_obj].append(c)
                        inputs[c.post_obj].append(c)

    return result_objs, result_conn


def find_all_io(connections):
    """Build up a list of all inputs and outputs for each object"""
    inputs = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    for c in connections:
        inputs[c.post_obj].append(c)
        outputs[c.pre_obj].append(c)
    return inputs, outputs
