def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    # t1, t2 = itertools.tee(iterable)
    # return itertools.ifilterfalse(pred, t1), filter(pred, t2)
    i1 = filter(pred, iterable)
    i2 = filter(lambda x: not pred(x), iterable)
    return i1, i2


def apply_feedforward(parent, cfg):
    """
    Arrange all items below parent into a feedforward layout.
    """
    sub_items = parent.children

    if len(sub_items) <= 0:
        return

    # recurse on items contained within the current item. After this loop
    # runs, all items at the current level should have had their sizes
    # appropriately set. The lower level networks have to be arranged in
    # feedforward order first, so that they have a definite size
    for item in sub_items:
        apply_feedforward(item, cfg)

    # Here do the "vertex" sorting. Not quite vertices, because
    # they have non-zero size (in the case of subnetworks) that we have to
    # take into account when positioning things

    def min_degree_partition(items, degree_func):
        # FIXME this if conditional can probably be removed
        if not items:
            return [], []

        min_degree = min([degree_func(item) for item in items])

        chosen_vertices, remaining_vertices = partition(
            lambda item: degree_func(item) == min_degree, items)

        return chosen_vertices, remaining_vertices

    in_degree_func = lambda v: len(v.incoming)
    out_degree_func = lambda v: len(v.outgoing)

    starting_vertices, _ = min_degree_partition(sub_items, in_degree_func)

    ending_vertices, _ = min_degree_partition(sub_items, out_degree_func)

    remaining_vertices = filter(
        lambda x: x not in starting_vertices and x not in ending_vertices,
        sub_items)

    intersection = filter(lambda x: x in ending_vertices, starting_vertices)

    for x in intersection:
        if len(x.outgoing) >= len(x.incoming):
            ending_vertices.remove(x)
        else:
            starting_vertices.remove(x)

    if not starting_vertices:
        starting_vertices, remaining_vertices = min_degree_partition(
            remaining_vertices, in_degree_func)

    layers = []
    if starting_vertices:
        layers.append(starting_vertices)

        while remaining_vertices:

            next_layer = []

            for v in layers[-1]:
                new_vertices, remaining_vertices = partition(
                    lambda nv: nv in v.outgoing,
                    remaining_vertices)

                next_layer.extend(new_vertices)

            if not next_layer:
                next_layer, remaining_vertices = min_degree_partition(
                    remaining_vertices, in_degree_func)

            layers.append(next_layer)

    if ending_vertices:
        layers.append(ending_vertices)

    # set the positions of the subitems accordingly
    h_spacing = 125
    h_subnet_padding = 50
    v_spacing = 75
    v_subnet_padding = 50

    h_offset = h_subnet_padding
    layer_heights = []
    layer_widths = [max([cfg[v.nengo_object].size[0] for v in layer])
                    for layer in layers]

    for i, layer in enumerate(layers):
        v_offset = v_subnet_padding

        for j, v in enumerate(layer):
            w, h = cfg[v.nengo_object].size
            # extra x term is for horizontal centering within layer
            cfg[v.nengo_object].pos = (
                h_offset + (layer_widths[i] - w) / 2.0,  # x
                v_offset)

            v_offset += h

            if j < len(layer) - 1:
                v_offset += v_spacing

        v_offset += v_subnet_padding
        layer_heights.append(v_offset)

        h_offset += layer_widths[i]

        if i < len(layers) - 1:
            h_offset += h_spacing

    max_height = max(layer_heights)
    cfg[parent.nengo_object].size = (h_offset + h_subnet_padding, max_height)

    # centre vertically
    for layer, height in zip(layers, layer_heights):

        v_offset = (max_height - height) / 2.0

        for v in layer:
            cfg[v.nengo_object].pos = (
                cfg[v.nengo_object].pos[0],
                cfg[v.nengo_object].pos[1] + v_offset)


def apply_transforms(parent, cfg):
    """ Propogate changes downward. Before this call, for all items,
    item.pos should be the position with respect to the network/item
    containing it. The purpose of this function is to make all item.pos
    the position of the item relative to the top-level network"""
    for v in parent.children:
        cfg[v.nengo_object].pos = (
            cfg[v.nengo_object].pos[0] + cfg[parent.nengo_object].pos[0],
            cfg[v.nengo_object].pos[1] + cfg[parent.nengo_object].pos[1])

        apply_transforms(v, cfg)


def feedforward_layout(model_graph, cfg):
    apply_feedforward(model_graph.top, cfg)
    apply_transforms(model_graph.top, cfg)
