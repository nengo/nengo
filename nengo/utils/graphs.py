"""Simple graph manipulation algorithms.

Nengo models are essentially graphs where ensembles, nodes, and networks
are graph vertices, and connections are edges. We make use of this fact
in some places in the code; this file contains functions to help with that.

----

toposort and reverse_edges are adapted from Theano (theano/gof/sched.py).
This modified code is included under the terms of their license:

Theano is copyright (c) 2008--2013 Theano Development Team.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Theano nor the names of its contributors may be
  used to endorse or promote products derived from this software without
 specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections import defaultdict

from ..exceptions import BuildError


def graph(edges=None):
    g = defaultdict(set)
    if edges is not None:
        g.update(edges)
    return g


class BidirectionalDAG:
    """Directed acyclic graph supporting bidirectional traversal.

    Parameters
    ----------
    forward : dict
        Forward edges for each vertex in the form
        {1: {2, 3}, 2: {3}, 3: set()}.

    Attributes
    ----------
    forward : dict
        Maps vertices to edges in forward direction.
    backward : dict
        Maps vertices to edges in backward direction.
    """

    def __init__(self, forward):
        self.forward = forward
        self.backward = reverse_edges(forward)

    def merge(self, vertices, merged_vertex):
        """Merges vertices in the graph.

        Parameters
        ----------
        vertices : set
            Vertices that are being merged.
        merged_vertex
            The vertex that replaces *vertices*.
        """

        forward_edges = set()
        for v in vertices:
            forward_edges.update(self.forward[v])
            del self.forward[v]
        self.forward[merged_vertex] = forward_edges

        backward_edges = set()
        for v in vertices:
            backward_edges.update(self.backward[v])
            del self.backward[v]
        self.backward[merged_vertex] = backward_edges

        for e in forward_edges:
            self.backward[e].difference_update(vertices)
            self.backward[e].add(merged_vertex)

        for e in backward_edges:
            self.forward[e].difference_update(vertices)
            self.forward[e].add(merged_vertex)


def toposort(edges):
    """Topological sort algorithm by Kahn [1]_.

    Complexity is O(nodes + vertices).

    Parameters
    ----------
    edges : dict
        Dict of the form {a: {b, c}} where b and c depend on a

    Returns
    -------
    An ordered list of nodes that satisfy the dependencies of ``edges``

    Example
    -------
    >>> toposort({1: {2, 3}, 2: {3}, 3: set()})
    [1, 2, 3]

    Notes
    -----
    Closely follows the wikipedia page [2]_.

    References
    ----------
    .. [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
       Communications of the ACM
    .. [2] https://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_edges(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    vertices = {v for v in edges if v not in incoming_edges or not incoming_edges[v]}
    ordered = []

    while vertices:
        n = vertices.pop()
        ordered.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                vertices.add(m)
    if any(incoming_edges.get(v, None) for v in edges):
        raise BuildError(
            "Input graph has cycles. This usually occurs because "
            "too many connections have no synapses. Try setting "
            "more synapses to '0' instead of 'None'."
        )
    return ordered


def transitive_closure(edges, topo_sorted=None):
    """Constructs the transitive closure of a directed acyclic graph (DAG).

    The complexity is O(nodes + vertices).

    Parameters
    ----------
    edges : dict
        Dict of the form ``{a: {b, c}}`` where ``b`` and ``c`` depend on ``a``.
        Must not contain cycles.
    topo_sorted : sequence, optional
        The topological sorting of the vertices. If not passed in, the
        algorithm will do a topological sort.

    Returns
    -------
    The transitive closure using the same data structure as `edges`: a dict
    of the form ``{a: {b, c}}`` where ``b`` and ``c`` are nodes that either
    directly or indirectly depend on ``a``.
    """
    if topo_sorted is None:
        topo_sorted = toposort(edges)

    sets = {}
    reachables = {}
    for vertex in reversed(topo_sorted):
        reachables[vertex] = set(edges[vertex])
        for edge in edges[vertex]:
            reachables[vertex].update(reachables[edge])
        reachables[vertex] = frozenset(reachables[vertex])

        # We try to reuse existing sets as this can significantly reduce
        # memory in some cases (which is important in the OpMergeOptimizer).
        if reachables[vertex] in sets:
            reachables[vertex] = sets[reachables[vertex]]
        sets[reachables[vertex]] = reachables[vertex]
    return reachables


def reverse_edges(edges):
    """Reverses direction of dependence dict.

    Parameters
    ----------
    edges : dict
        Dict of the form {a: {b, c}, b: set(), c: set()} where b and c depend
        on a.

    Returns
    -------
    Dict of the form {a: set(), b: {a}, c: {a}} where b and c depend on a.

    Example
    -------

    >>> d = {0: {1, 2}, 1: {2, 3}, 2: set(), 3: set()}
    >>> reverse_edges(d)
    {0: set(), 1: {0}, 2: {0, 1}, 3: {1}}

    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.

    """
    result = {k: set() for k in edges}
    for key in edges:
        for val in edges[key]:
            result[val].add(key)
    return result


def add_edges(edges, new_edge_iter):
    for src, dst in new_edge_iter:
        edges[src].add(dst)
