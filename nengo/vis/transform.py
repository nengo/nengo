from nengo.vis.modelgraph import CollapsedConnection, CollapsedNetwork


# TODO unit test
def collapse(model_graph, v_net):
    incoming = []
    outgoing = []
    for v in list(v_net.descendants):
        incoming.extend(e for e in v.incoming if e.parent in v_net.ancestors)
        outgoing.extend(e for e in v.outgoing if e.parent in v_net.ancestors)
        model_graph.remove_vertex(v)
    parent = v_net.parent

    model_graph.remove_vertex(v_net)

    v_collapsed = CollapsedNetwork(v_net.nengo_object, v_net)
    model_graph.add_vertex(v_collapsed, parent=parent)
    for v in incoming:
        model_graph.add_edge(CollapsedConnection(v, v_collapsed))
    for v in outgoing:
        model_graph.add_edge(CollapsedConnection(v_collapsed, v))
