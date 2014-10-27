import pytest

import nengo
from nengo.vis.modelgraph import ModelGraph


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

    def test_top_attribute(self):
        model1 = nengo.Network()
        with model1:
            subnet = nengo.Network()
        model2 = nengo.Network()

        g = ModelGraph()
        g.add_network(model1)
        g.add_network(model2)

        assert g.top.nengo_object == model1


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
