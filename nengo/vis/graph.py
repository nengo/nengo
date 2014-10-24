

class Vertex(object):
    __slots__ = ['nengo_object', 'parent', 'children', 'incoming', 'outgoing']

    def __init__(self, nengo_object, parent=None, children=None):
        self.nengo_object = nengo_object
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

    @property
    def descendants(self):
        stack = list(self.children)
        descendants = []
        while len(stack) > 0:
            descendants.append(stack.pop())
            stack.extend(descendants[-1].descendants)
        return descendants


class Node(Vertex):
    def is_input(self):
        return self.nengo_object.size_out > 0

    def is_output(self):
        return self.nengo_object.size_in > 0

    def is_pure_input(self):
        return self.is_input() and not self.is_output()

    def is_pure_output(self):
        return not self.is_input() and self.is_output()


class Ensemble(Vertex):
    pass


class Network(Vertex):
    pass


class CollapsedNetwork(Vertex):
    __slots__ = ['collapsed']

    def __init__(self, nengo_object, collapsed, parent=None, children=[]):
        super(CollapsedNetwork, self).__init__(
            nengo_object, parent=parent, children=children)
        self.collapsed = collapsed


class Graph(object):
    def __init__(self):
        pass

    def add_vertex(self, v):
        pass

    def add_edge(self, source, target):
        source.outgoing.append(target)
        target.incoming.append(source)
