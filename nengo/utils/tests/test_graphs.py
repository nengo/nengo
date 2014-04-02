import logging

import pytest

import nengo
from nengo.utils import graphs

logger = logging.getLogger(__name__)


def test_reversedict():
    edges = graphs.graph({'a': set(['b', 'c'])})
    r_edges = graphs.reverse_edges(edges)
    assert r_edges == {'b': ('a',), 'c': ('a',)}


def test_toposort():
    edges = graphs.graph({'a': set(['b', 'c']), 'b': ('c',)})
    assert graphs.toposort(edges) == ['a', 'b', 'c']


def test_add_edges():
    edges = graphs.graph({'a': set(['b', 'c'])})
    graphs.add_edges(edges, [('a', 'd'), ('b', 'c')])
    assert edges == {'a': set(['b', 'c', 'd']), 'b': set(['c'])}


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
