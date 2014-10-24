import nengo


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
        self._nengo_object_to_vertex = {}

    def add_vertex(self, v, parent=None):
        self._nengo_object_to_vertex[v.nengo_object] = v
        if parent is not None:
            v.parent = parent
            parent.children.append(v)

    def add_edge(self, source, target):
        source.outgoing.append(target)
        target.incoming.append(source)

    def get_vertex(self, nengo_object):
        return self._nengo_object_to_vertex[nengo_object]


class ModelGraph(Graph):
    def __init__(self, model=None):
        super(ModelGraph, self).__init__()
        if model is not None:
            self.add_network(model)

    def add_network(self, net):
        v_net = Network(net)
        self.add_vertex(v_net)

        self._add_objects_with_conversion(
            net.ensembles, Ensemble, parent=v_net)
        self._add_objects_with_conversion(net.nodes, Node, parent=v_net)
        self._add_objects_with_conversion(net.networks, Network, parent=v_net)

        for conn in net.connections:
            pre = conn.pre_obj
            if isinstance(pre, nengo.ensemble.Neurons):
                pre = pre.ensemble
            post = conn.post_obj
            if isinstance(post, nengo.ensemble.Neurons):
                post = post.ensemble
            self.add_edge(self.get_vertex(pre), self.get_vertex(post))

    def _add_objects_with_conversion(self, objects, conversion, parent=None):
        for obj in objects:
            self.add_vertex(conversion(obj), parent=parent)
