# import itertools


def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    # t1, t2 = itertools.tee(iterable)
    # return itertools.ifilterfalse(pred, t1), filter(pred, t2)
    i1 = filter(pred, iterable)
    i2 = filter(lambda x: not pred(x), iterable)
    return i1, i2


class Item(object):
    depth = 0
    delim = '-> '

    def __init__(self, node):
        self.node = node
        self.members = []
        self.parent = None

        self.adjacency_matrix = []
        self._in_degree = []
        self._out_degree = []

        self.pos = [0.0, 0.0]

        # should eventually incorporate ability to keep some nodes fixed
        self.fixed = False

        if node is not None:
            self.obj_id = node['id']
            self.width = node.get('width', 0.0)
            self.height = node.get('height', 0.0)

            self.top_level = False
        else:
            self.top_level = True

    @staticmethod
    def lowest_common_parent(item1, item2):
        """Return None if no common parents. Otherwise, return the lowest
        common parent, along with the two nodes that item1 and item2
        are a part of in that common parent."""

        parents1 = item1.all_parents()
        parents2 = item2.all_parents()

        for p in parents1:
            if p in parents2:
                index = parents1.index(p)
                if index > 0:
                    item1 = parents1[index - 1]

                index = parents2.index(p)
                if index > 0:
                    item2 = parents2[index - 1]

                return p, item1, item2

        return None, None, None

    def all_parents(self):
        """From most immediate to least immediate"""
        item = self
        parents = []

        while(item.parent is not None):
            item = item.parent
            parents.append(item)

        return parents

    def add_subitem(self, item):
        item.index_in_parent = len(self.members)

        self.members.append(item)

        item.parent = self
        self.adjacency_matrix.append([False] * len(self.members))
        for adj in self.adjacency_matrix:
            adj.append(False)

        self._in_degree.append(0)
        self._out_degree.append(0)

    def add_link(self, source, target):
        if source not in self.members:
            raise ValueError("Failed to add link: source item %s not a member"
                             " of this item" % str(source))
        if target not in self.members:
            raise ValueError("Failed to add link: target item %s not a member"
                             " of this item" % str(target))

        i = source.index_in_parent
        j = target.index_in_parent

        self.adjacency_matrix[i][j] = True
        self._in_degree[j] += 1
        self._out_degree[i] += 1

    def link_exists(self, source, target):
        if source not in self.members:
            raise ValueError("Failed to check link: source item %s not a"
                             " member of this item" % str(source))
        if target not in self.members:
            raise ValueError("Failed to check link: target item %s not a"
                             " member of this item" % str(target))

        i = source.index_in_parent
        j = target.index_in_parent
        return self.adjacency_matrix[i][j]

    def out_degree(self, item):
        if item not in self.members:
            raise ValueError("Failed to get out_degree: item %s not a member"
                             " of this item" % str(item))

        return self._out_degree[item.index_in_parent]

    def in_degree(self, item):
        if item not in self.members:
            raise ValueError("Failed to get in_degree: item %s not a member"
                             " of this item" % str(item))

        return self._in_degree[item.index_in_parent]

    def __str__(self):
        ret = ['', 'GraphItem{']
        for attr in dir(self):
            if attr[0:2] != '__' and attr != 'members' and attr != 'parent':
                val = getattr(self, attr)
                if not callable(val):
                    ret.append(attr+': '+str(getattr(self, attr)))

        Item.depth += 1
        ret.append("Members:")
        for item in self.members:
            ret.append(item.__str__())
        Item.depth -= 1

        ret.append('}')

        ret = ('\n'+(Item.delim*Item.depth)).join(ret)
        return ret


def convert(links, nodes):
    """
    Convert the dictionary representation of a nengo network
    (as stored in an instance of nengo_gui.convert.Converter)
    and return the corresponding hierarchical graph in terms of items
    and links. Importantly, in the resulting graph all connections are
    stored at the level of the lowest common parent of the connected nodes
    """

    num_nodes = len(nodes)

    items = []
    for node in nodes:
        item = Item(node)
        items.append(item)

    for node, item in zip(nodes, items):
        if 'contains' in node:
            for c in node['contains']:
                item.add_subitem(items[c])

    top_level = filter(
        lambda i: nodes[i]['contained_by'] == -1, range(num_nodes))
    top_level = map(lambda i: items[i], top_level)

    top_item = Item(None)

    for item in top_level:
        top_item.add_subitem(item)

    for link in links:
        source_item = items[link['source']]
        target_item = items[link['target']]

        if source_item != target_item:
            # self loops have no bearing on layout
            common_parent, source_item, target_item = \
                Item.lowest_common_parent(source_item, target_item)

            common_parent.add_link(source_item, target_item)

    return top_item


def apply_feedforward(parent):
    """
    Arrange all items below parent into a feedforward layout.
    """
    sub_items = parent.members

    if not sub_items:
        return

    # recurse on items contained within the current item. After this loop
    # runs, all items at the current level should have had their sizes
    # appropriately set. The lower level networks have to be arranged in
    # feedforward order first, so that they have a definite size
    for item in sub_items:
        apply_feedforward(item)

    # Here do the "vertex" sorting. Not quite vertices, because
    # they have non-zero size (in the case of subnetworks) that we have to
    # take into account when positioning things

    def min_degree_partition(parent, items, degree_func):
        if not items:
            return [], []

        min_degree = min([degree_func(parent, item) for item in items])

        chosen_vertices, remaining_vertices = partition(
            lambda item: degree_func(parent, item) == min_degree, items)

        return chosen_vertices, remaining_vertices

    in_degree_func = lambda p, i: p.in_degree(i)
    out_degree_func = lambda p, i: p.out_degree(i)

    starting_vertices, _ = min_degree_partition(
        parent, sub_items, in_degree_func)

    ending_vertices, _ = min_degree_partition(
        parent, sub_items, out_degree_func)

    remaining_vertices = filter(
        lambda x: x not in starting_vertices and x not in ending_vertices,
        sub_items)

    intersection = filter(lambda x: x in ending_vertices, starting_vertices)

    for x in intersection:
        if parent.out_degree(x) >= parent.in_degree(x):
            ending_vertices.remove(x)
        else:
            starting_vertices.remove(x)

    if not starting_vertices:
        starting_vertices, remaining_vertices = min_degree_partition(
            parent, remaining_vertices, in_degree_func)

    layers = []
    if starting_vertices:
        layers.append(starting_vertices)

        while remaining_vertices:

            next_layer = []

            for v in layers[-1]:
                new_vertices, remaining_vertices = partition(
                    lambda nv: parent.link_exists(v, nv),
                    remaining_vertices)

                next_layer.extend(new_vertices)

            if not next_layer:
                next_layer, remaining_vertices = min_degree_partition(
                    parent, remaining_vertices, in_degree_func)

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
    layer_widths = [max([item.width for item in layer]) for layer in layers]

    for i, layer in enumerate(layers):
        v_offset = v_subnet_padding

        for j, item in enumerate(layer):
            # extra x term is for horizontal centering within layer
            item.pos[0] = h_offset + (layer_widths[i] - item.width) / 2.0  # x
            item.pos[1] = v_offset

            v_offset += item.height

            if j < len(layer) - 1:
                v_offset += v_spacing

        v_offset += v_subnet_padding
        layer_heights.append(v_offset)

        h_offset += layer_widths[i]

        if i < len(layers) - 1:
            h_offset += h_spacing

    parent.width = h_offset + h_subnet_padding

    max_height = max(layer_heights)
    parent.height = max_height

    # centre vertically
    for layer, height in zip(layers, layer_heights):

        v_offset = (max_height - height) / 2.0

        for item in layer:
            item.pos[1] += v_offset


def apply_transforms(parent):
    """ Propogate changes downward. Before this call, for all items,
    item.pos should be the position with respect to the network/item
    containing it. The purpose of this function is to make all item.pos
    the position of the item relative to the top-level network"""
    for item in parent.members:
        item.pos[0] += parent.pos[0]
        item.pos[1] += parent.pos[1]

        apply_transforms(item)


def populate_config(parent, config):
    """Store the computed layput in the config and in the dictionaries
    that get converted to json (ie item.node)"""
    for item in parent.members:
        if item.members:
            populate_config(item, config)

            # this is so that the javascript engine doesn't
            # move things around when rendering the layout
            # see update_node_positions in static/js/main.js
            if 'x' in item.node:
                del item.node['x']
            if 'y' in item.node:
                del item.node['y']
            if 'width' in item.node:
                del item.node['width']
            if 'height' in item.node:
                del item.node['height']
        else:
            key = get_obj_for_id(item.obj_id)
            config[key].pos = item.pos
            item.node['x'] = item.pos[0]
            item.node['y'] = item.pos[1]


# FIXME this is an extremely horrible hack
import gc


def get_obj_for_id(id_):
    for obj in gc.get_objects():
        if id(obj) == id_:
            return obj
    raise KeyError()


def feedforward_layout(network, config, links, nodes):
    top_level_item = convert(links, nodes)
    apply_feedforward(top_level_item)
    apply_transforms(top_level_item)
    populate_config(top_level_item, config)
