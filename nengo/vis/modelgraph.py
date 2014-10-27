import nengo
from nengo.vis.graph import Edge, Graph, Vertex


class NengoVertex(Vertex):
    __slots__ = ['nengo_object']

    def __init__(self, nengo_object, parent=None, children=None):
        super(NengoVertex, self).__init__(parent=parent, children=children)
        self.nengo_object = nengo_object


class Node(NengoVertex):
    def is_input(self):
        return self.nengo_object.size_out > 0

    def is_output(self):
        return self.nengo_object.size_in > 0

    def is_pure_input(self):
        return self.is_input() and not self.is_output()

    def is_pure_output(self):
        return not self.is_input() and self.is_output()


class Ensemble(NengoVertex):
    pass


class Network(NengoVertex):
    pass


class CollapsedNetwork(NengoVertex):
    __slots__ = ['collapsed']

    def __init__(self, nengo_object, collapsed, parent=None, children=[]):
        super(CollapsedNetwork, self).__init__(
            nengo_object, parent=parent, children=children)
        self.collapsed = collapsed


class Connection(Edge):
    __slots__ = ['nengo_object']

    def __init__(self, source, target, nengo_object):
        super(Connection, self).__init__(source, target)
        self.nengo_object = nengo_object


class ModelGraph(Graph):
    def __init__(self, model=None):
        super(ModelGraph, self).__init__()
        self._nengo_object_to_vertex = {}
        self.top = None
        if model is not None:
            self.add_network(model)

    def add_vertex(self, v, parent=None):
        self._nengo_object_to_vertex[v.nengo_object] = v
        super(ModelGraph, self).add_vertex(v, parent=parent)

    def add_network(self, net):
        v_net = Network(net)
        self.add_vertex(v_net)

        if self.top is None:
            self.top = v_net

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
            self.add_edge(Connection(
                self.get_vertex(pre), self.get_vertex(post), conn))

    def _add_objects_with_conversion(self, objects, conversion, parent=None):
        for obj in objects:
            self.add_vertex(conversion(obj), parent=parent)

    def get_vertex(self, nengo_object):
        return self._nengo_object_to_vertex[nengo_object]


class Renderer(object):
    def render_vertex(self, v):
        call_table = {
            Ensemble: self.render_ensemble,
            Node: self.render_node,
            Network: self.render_network,
            CollapsedNetwork: self.render_network,
        }
        return call_table[v.__class__](v)

    def render_ensemble(self, ens):
        raise NotImplementedError()

    def render_node(self, node):
        raise NotImplementedError()

    def render_network(self, net):
        raise NotImplementedError()

    def render_collapsed_network(self, cnet):
        raise NotImplementedError()
