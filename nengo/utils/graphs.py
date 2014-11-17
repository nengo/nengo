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

from .compat import iteritems


def graph(edges=None):
    g = defaultdict(set)
    if edges is not None:
        g.update(edges)
    return g


def toposort(edges):
    """Topological sort algorithm by Kahn[1]

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

    >>> toposort({1: {2, 3}, 2: (3,)})
    [1, 2, 3]

    Notes
    -----

    Closely follows the wikipedia page [2]

    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_edges(edges)
    incoming_edges = dict((k, set(val))
                          for k, val in iteritems(incoming_edges))
    vertices = set((v for v in edges if v not in incoming_edges))
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
        raise ValueError("Input graph has cycles. This usually occurs because "
                         "too many connections have no synapses. Try setting "
                         "more synapses to '0' instead of 'None'.")
    return ordered


def reverse_edges(edges):
    """Reverses direction of dependence dict.

    Parameters
    ----------
    edges : dict
        Dict of the form {a: {b, c}} where b and c depend on a

    Returns
    -------
    Dict of the form {b: (a,), c: (a,)} where b and c depend on a

    Example
    -------

    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_edges(d)
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.

    """
    result = {}
    for key in edges:
        for val in edges[key]:
            result[val] = result.get(val, tuple()) + (key, )
    return result


def add_edges(edges, new_edge_iter):
    for src, dst in new_edge_iter:
        edges[src].add(dst)
