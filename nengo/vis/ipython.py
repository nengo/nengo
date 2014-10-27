import json
import pkgutil
import uuid

from IPython.display import display, HTML

from nengo.vis.modelgraph import Renderer


# TODO move somewhere else
class Identificator(object):
    def get_id(self, obj):
        raise NotImplementedError()


class SimpleIdentificator(Identificator):
    def get_id(self, obj):
        return id(obj)


class ModelGraphDisplay(object):
    def __init__(self, data):
        self.data = data

    def _ipython_display_(self):
        js = pkgutil.get_data('nengo.vis', 'static/js/main.js')
        d3 = pkgutil.get_data('nengo.vis', 'static/js/vendor/d3.min.js')
        css = pkgutil.get_data('nengo.vis', 'static/css/graph.css')
        display(HTML('''
            <script>
                if (typeof(d3) == 'undefined') {{
                    {d3}
                }}
            </script>
            <style type="text/css">{css}</style>
            <script>
                {js}
                main({data});
            </script>
            <div id="#graph"><svg>
                <defs>
                    // the arrow on the links
                    <marker id="TriangleMarker"
                            viewBox="0 0 10 10" refX="0" refY="5"
                            markerUnits="strokeWidth"
                            markerWidth="6" markerHeight="4"
                            orient="auto">
                        <path d="M 0 0 L 10 5 L 0 10 z" />
                    </marker>
                    <g id ="ensemble" transform="translate(-17.2735,-17.7015)">
                        <circle id="mainCircle" cx="16" cy="18" r="18"/>
                        <circle cx="4.843" cy="10.519" r="4.843"/>
                        <circle cx="16.186" cy="17.873" r="4.843"/>
                        <circle cx="21.012" cy="30.56" r="4.843"/>
                        <circle cx="29.704" cy="17.229" r="4.843"/>
                        <circle cx="5.647" cy="26.413" r="4.843"/>
                        <circle cx="19.894" cy="4.842" r="4.843"/>
                    </g>
                    <g id = "recur">
                        <path
                            d="M6.451,28.748C2.448,26.041,0,22.413,0,18.425C0,
                            10.051,10.801,3.262,24.125,3.262
                            S48.25,10.051,48.25,18.425c0,6.453-6.412,11.964-15.45,14.153"/>
                    </g>
                    <g>
                        <path id = "recurTriangle"
                        d="M 8 0 L 0 4 L 8 8 z""/>
                    </g>
                </defs>
            </svg></div>'''.format(
                js=js, d3=d3, css=css, data=self.data)))


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

        global_scale = self.cfg[model_graph.top.nengo_object].scale
        global_offset = self.cfg[model_graph.top.nengo_object].offset

        data = dict(
            nodes=vertices, links=edges,
            global_scale=global_scale, global_offset=global_offset)
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
        size = self.cfg[net.nengo_object].size
        return {
            'type': 'net',
            'contains': [self._vertex_to_index[v] for v in net.children],
            'full_contains': [
                self._vertex_to_index[v] for v in net.descendants],
            'width': size[0], 'height': size[1],
        }

    def render_connection(self, conn):
        pre_idx = self._vertex_to_index[conn.source]
        post_idx = self._vertex_to_index[conn.target]
        if pre_idx == post_idx:
            connection_type = 'rec'
        else:
            connection_type = 'std'
        if hasattr(conn, 'nengo_object'):
            conn_id = self.identificator.get_id(conn.nengo_object)
        else:
            conn_id = str(uuid.uuid4())
        return {
            'source': pre_idx,
            'target': post_idx,
            'id': conn_id,
            'type': connection_type
        }
