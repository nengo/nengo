import json
import pprint

import nengo


class Vertex(object):
    __slots__ = ['parent', 'children', 'incoming', 'outgoing']

    def __init__(self, parent=None, children=None):
        self.parent = parent
        if children is None:
            children = []
        self.children = children
        self.incoming = []
        self.outgoing = []

    @property
    def ancestors(self):
        ancestors = [self.parent]
        while ancestors[-1].parent is not None:
            ancestors.append(ancestors[-1].parent)
        return ancestors

    # TODO unit test with repeated descendants (elements of returned list/set
    # should be unique)
    @property
    def descendants(self):
        stack = list(self.children)
        stacks = []
        descendants = []
        while len(stack) > 0:
            descendants.append(stack.pop())
            stacks.append(list(descendants))
            stack.extend(descendants[-1].descendants)
        return set(descendants)


class Edge(object):
    __slots__ = ['source', 'target']

    def __init__(self, source, target):
        self.source = source
        self.target = target


class Graph(object):
    def __init__(self):
        self.vertices = []
        self.edges = []

    def add_vertex(self, v, parent=None):
        if parent is not None:
            v.parent = parent
            parent.children.append(v)
        self.vertices.append(v)

    def add_edge(self, e):
        e.source.outgoing.append(e.target)
        e.target.incoming.append(e.source)
        self.edges.append(e)

    # TODO unit test
    def remove_edge(self, e):
        e.source.outgoing.remove(e.target)
        e.target.incoming.remove(e.source)
        self.edges.remove(e)

    # TODO unit test
    def remove_vertex(self, v):
        for e in list(self.edges):
            if e.source is v or e.target is v:
                self.remove_edge(e)
            elif e.source in v.descendants or e.target in v.descendants:
                self.remove_edge(e)
        if v.parent is not None:
            v.parent.children.remove(v)
        v.parent = None
        self.vertices.remove(v)
        for c in v.descendants:
            self.vertices.remove(c)
