import json
import pprint

import nengo


# TODO move somewhere else
class Identificator(object):
    def get_id(self, obj):
        raise NotImplementedError()


class SimpleIdentificator(Identificator):
    def get_id(self, obj):
        return id(obj)


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

    @property
    def descendants(self):
        stack = list(self.children)
        descendants = []
        while len(stack) > 0:
            descendants.append(stack.pop())
            stack.extend(descendants[-1].descendants)
        return descendants


class Edge(object):
    __slots__ = ['source', 'target']

    def __init__(self, source, target):
        self.source = source
        self.target = target


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


class Graph(object):
    def __init__(self):
        self.vertices = []
        self.edges = []  # TODO test

    def add_vertex(self, v, parent=None):
        if parent is not None:
            v.parent = parent
            parent.children.append(v)
        self.vertices.append(v)

    def add_edge(self, e):
        e.source.outgoing.append(e.target)
        e.target.incoming.append(e.source)
        self.edges.append(e)


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
        if self.top is None:
            self.top = net  # TODO test top

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


class D3DataRenderer(Renderer):
    def __init__(self, cfg, identificator=SimpleIdentificator()):
        self.cfg = cfg
        self.identificator = identificator
        self._vertex_to_index = {}

    def render(self, model_graph):
        for i, v in enumerate(model_graph.vertices):
            self._vertex_to_index[v] = i
        vertices = [self.render_vertex(v) for v in model_graph.vertices]
        edges = [self.render_connection(e) for e in model_graph.edges]

        global_scale = self.cfg[model_graph.top].scale
        global_offset = self.cfg[model_graph.top].offset

        data = dict(
            nodes=vertices, links=edges,
            global_scale=global_scale, global_offset=global_offset)
        pprint.pprint(data)
        return json.dumps(data)

    def render_vertex(self, v):
        pos = self.cfg[v.nengo_object].pos
        scale = self.cfg[v.nengo_object].scale

        if v.parent is None:
            contained_by = -1
        else:
            contained_by = self._vertex_to_index[v.parent]

        data = super(D3DataRenderer, self).render_vertex(v)
        data.update({
            'label': v.nengo_object.label,
            'id': self.identificator.get_id(v.nengo_object),
            'x': pos[0], 'y': pos[1], 'scale': scale,
            'contained_by': contained_by,
        })
        return data

    def render_ensemble(self, ens):
        return {'type': 'ens'}

    def render_node(self, node):
        return {'type': 'nde', 'is_input': node.is_pure_input()}

    def render_network(self, net):
        size = self.cfg[net.nengo_object].size
        return {
            'type': 'net',
            'contains': [self._vertex_to_index[v] for v in net.children],
            'full_contains': [
                self._vertex_to_index[v] for v in net.descendants],
            'width': size[0], 'height': size[1],
        }

    def render_collapsed_network(self, cnet):
        raise NotImplementedError()

    def render_connection(self, conn):
        pre_idx = self._vertex_to_index[conn.source]
        post_idx = self._vertex_to_index[conn.target]
        if pre_idx == post_idx:
            connection_type = 'rec'
        else:
            connection_type = 'std'
        return {
            'source': pre_idx,
            'target': post_idx,
            'id': self.identificator.get_id(conn.nengo_object),
            'type': connection_type
        }
