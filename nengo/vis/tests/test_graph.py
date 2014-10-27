import pytest

import nengo
from nengo.vis.graph import Edge, Graph, Vertex


class TestVertex(object):
    def test_ancestors(self):
        v1 = Vertex()
        v2a = Vertex(parent=v1)
        v2b = Vertex(parent=v1)
        v3 = Vertex(parent=v2a)
        assert v3.ancestors == [v2a, v1]

    def test_descendants(self):
        v1 = Vertex()
        v2a = Vertex()
        v2b = Vertex()
        v3 = Vertex()
        v1.children.append(v2a)
        v1.children.append(v2b)
        v2a.children.append(v3)
        # Sort lists because we do not care about order
        assert sorted(v1.descendants) == sorted([v2a, v2b, v3])


class TestGraph(object):
    def test_build_graph(self):
        vertices = [Vertex() for i in range(4)]

        # Construct this graph:
        # 0 --+--> 1 ---+--> 3
        #     |         |
        #     +--> 2 <--+
        g = Graph()
        for v in vertices:
            g.add_vertex(v)
        g.add_edge(Edge(vertices[0], vertices[1]))
        g.add_edge(Edge(vertices[0], vertices[2]))
        g.add_edge(Edge(vertices[1], vertices[2]))
        g.add_edge(Edge(vertices[1], vertices[3]))

        assert sorted(g.vertices) == sorted(vertices)
        assert vertices[0].incoming == []
        assert vertices[0].outgoing == [vertices[1], vertices[2]]
        assert vertices[1].incoming == [vertices[0]]
        assert vertices[1].outgoing == [vertices[2], vertices[3]]
        assert vertices[2].incoming == [vertices[0], vertices[1]]
        assert vertices[2].outgoing == []
        assert vertices[3].incoming == [vertices[1]]
        assert vertices[3].outgoing == []

    def test_build_parent_relation(self):
        v1 = Vertex()
        v2 = Vertex()

        g = Graph()
        g.add_vertex(v1)
        g.add_vertex(v2, parent=v1)

        assert v1.children == [v2]


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
