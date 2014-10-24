import pytest

import nengo
from nengo.vis.graph import Graph, ModelGraph, Vertex


class TestVertex(object):
    def test_ancestors(self):
        v1 = Vertex(None)
        v2a = Vertex(None, parent=v1)
        v2b = Vertex(None, parent=v1)
        v3 = Vertex(None, parent=v2a)
        assert v3.ancestors == [v2a, v1]

    def test_descendants(self):
        v1 = Vertex(None)
        v2a = Vertex(None)
        v2b = Vertex(None)
        v3 = Vertex(None)
        v1.children.append(v2a)
        v1.children.append(v2b)
        v2a.children.append(v3)
        # Sort lists because we do not care about order
        assert sorted(v1.descendants) == sorted([v2a, v2b, v3])


class TestGraph(object):
    def test_build_graph(self):
        vertices = [Vertex(None) for i in range(4)]

        # Construct this graph:
        # 0 --+--> 1 ---+--> 3
        #     |         |
        #     +--> 2 <--+
        g = Graph()
        for v in vertices:
            g.add_vertex(v)
        g.add_edge(vertices[0], vertices[1])
        g.add_edge(vertices[0], vertices[2])
        g.add_edge(vertices[1], vertices[2])
        g.add_edge(vertices[1], vertices[3])

        assert vertices[0].incoming == []
        assert vertices[0].outgoing == [vertices[1], vertices[2]]
        assert vertices[1].incoming == [vertices[0]]
        assert vertices[1].outgoing == [vertices[2], vertices[3]]
        assert vertices[2].incoming == [vertices[0], vertices[1]]
        assert vertices[2].outgoing == []
        assert vertices[3].incoming == [vertices[1]]
        assert vertices[3].outgoing == []

    def test_build_parent_relation(self):
        v1 = Vertex(None)
        v2 = Vertex(None)

        g = Graph()
        g.add_vertex(v1)
        g.add_vertex(v2, parent=v1)

        assert v1.children == [v2]


class TestModelGraph(object):
    def test_build_graph(self):
        model = nengo.Network()
        with model:
            a = nengo.Node(output=1)
            b = nengo.Ensemble(10, 1)
            c = nengo.Ensemble(10, 1)
            d = nengo.Ensemble(10, 1)
            nengo.Connection(a, b)
            nengo.Connection(b, c)
            nengo.Connection(a, d)
            nengo.Connection(b, d)

        g = ModelGraph(model)

        v_net = g.get_vertex(model)
        v_a = g.get_vertex(a)
        v_b = g.get_vertex(b)
        v_c = g.get_vertex(c)
        v_d = g.get_vertex(d)

        assert sorted(v_net.descendants) == sorted([v_a, v_b, v_c, v_d])
        assert v_a.parent == v_net
        assert v_a.incoming == []
        assert v_a.outgoing == [v_b, v_d]
        assert v_b.parent == v_net
        assert v_b.incoming == [v_a]
        assert v_b.outgoing == [v_c, v_d]
        assert v_c.parent == v_net
        assert v_c.incoming == [v_b]
        assert v_c.outgoing == []
        assert v_d.parent == v_net
        assert v_d.incoming == [v_a, v_b]
        assert v_d.outgoing == []


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
