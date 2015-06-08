import json
import re
import keyword
import namefinder

import pprint
import nengo


def isidentifier(s):
    if s in keyword.kwlist:
        return False
    return re.match(r'^[a-z_][a-z0-9_]*$', s, re.I) is not None


class Converter(object):
    def __init__(self, model, codelines, locals, config):
        self.model = model
        self.namefinder = namefinder.NameFinder(locals, model)
        self.codelines = codelines
        self.objects = []
        self.config = config
        self.links = []
        self.object_index = {model: -1}
        self.process(model)

        self.global_scale = config[model].scale
        self.global_offset = config[model].offset

    def process(self, network, id_prefix=None):
        """Process the network, ensembles, nodes, connections and probes into 
        a dictionary format to be processed by 
        the client side browser for visualization."""

        for i, ens in enumerate(network.ensembles):
            line = ens._created_line_number-1
            label = ens.label
            ens_id = self.namefinder.name(ens)

            pos = self.config[ens].pos
            scale = self.config[ens].scale

            if pos is None:
                pos = 0, 0

            if scale is None:
                scale = 1

            obj = {'label': label, 'line': line, 'id': ens_id, 'type': 'ens',
                   'x': pos[0], 'y': pos[1], 'scale': scale,
                   'contained_by': self.object_index[network]}

            self.object_index[ens] = len(self.objects)
            self.objects.append(obj)

        for i, nde in enumerate(network.nodes):
            line = nde._created_line_number-1
            label = nde.label
            if label == 'Node':
                label = ''
            nde_id = self.namefinder.name(nde)

            pos = self.config[nde].pos
            scale = self.config[nde].scale

            if pos is None:
                pos = 0, 0

            if scale is None:
                scale = 1

            is_input = (nde.size_in == 0 and nde.size_out >= 1)

            obj = {'label':label, 'line':line, 'id':nde_id, 'type':'nde',
                   'x':pos[0], 'y':pos[1],  'scale': scale,
                   'contained_by': self.object_index[network], "is_input": is_input}
            self.object_index[nde] = len(self.objects)
            self.objects.append(obj)


        full_contains={}
        for i, net in enumerate(network.networks):
            if not hasattr(net, '_created_line_number'):
                for obj in net.ensembles + net.nodes + net.connections:
                    net._created_line_number = obj._created_line_number
                    break
                else:
                    net._created_line_number = 0
            line = net._created_line_number-1
            label = net.label
            net_id = self.namefinder.name(net)

            self.object_index[net] = len(self.objects)
            self.objects.append({'placeholder':0}) # place holder

            # recursive call to process all sub-networks
            full_contains[i] = self.process(net, id_prefix=net_id)

            contains = [self.object_index[obj] for obj in
                net.ensembles + net.nodes + net.networks]

            full_contains[i] += contains

            pos = self.config[net].pos
            scale = self.config[net].scale
            size = self.config[net].size

            if pos is None:
                pos = -50, -50

            if scale is None:
                scale = 1

            if size is None:
                size = 100, 100

            obj = {'label':label, 'line':line, 'id':net_id, 'type':'net',
                   'contains':list(contains), 'full_contains': list(full_contains[i]),
                   'contained_by': self.object_index[network], 'scale': scale,
                   'x':pos[0], 'y':pos[1], 'width':size[0], 'height':size[1]}
            self.objects[self.object_index[net]] = obj

        for i, conn in enumerate(network.connections):
            conn_id = self.namefinder.name(conn)
            pre = conn.pre_obj
            if isinstance(pre, nengo.ensemble.Neurons):
                pre = pre.ensemble
            post = conn.post_obj
            if isinstance(post, nengo.ensemble.Neurons):
                post = post.ensemble
            # TODO: have a visual indication of direct connections
            if self.object_index[pre] == self.object_index[post]:
                connection_type = 'rec'
            else:
                connection_type = 'std'
            self.links.append({'source':self.object_index[pre],
                               'target':self.object_index[post],
                               'id':conn_id,
                               'type':connection_type})

        return sum(full_contains.values(),[])

    def to_json(self):
        data = dict(nodes=self.objects, links=self.links,
                    global_scale=self.global_scale,
                    global_offset=self.global_offset)
        pprint.pprint(data)
        return json.dumps(data)
