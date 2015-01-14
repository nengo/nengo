from nengo.vis.modelgraph import CollapsedConnection, CollapsedNetwork


# TODO unit test
def collapse(model_graph, v_net):
    incoming = []
    outgoing = []
    for v in list(v_net.descendants):
        incoming.extend(e for e in v.incoming if e not in v_net.descendants)
        outgoing.extend(e for e in v.outgoing if e not in v_net.descendants)
    parent = v_net.parent

    model_graph.remove_vertex(v_net)

    v_collapsed = CollapsedNetwork(v_net.nengo_object, v_net)
    model_graph.add_vertex(v_collapsed, parent=parent)
    for v in incoming:
        model_graph.add_edge(CollapsedConnection(v, v_collapsed))
    for v in outgoing:
        model_graph.add_edge(CollapsedConnection(v_collapsed, v))


# TODO unit test
def prune_edges(model_graph):
    connections = []
    for i, e in enumerate(list(model_graph.edges)):
        if (e.source, e.target) in connections:
            model_graph.remove_edge(e)
        else:
            connections.append((e.source, e.target))


# TODO unit test
def prune_empty_and_unconnected(model_graph):
    for v in list(model_graph.vertices):
        if (
                len(v.children) <= 0 and len(v.incoming) <= 0 and
                len(v.outgoing) <= 0):
            model_graph.remove_vertex(v)
