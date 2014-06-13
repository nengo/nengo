import socket
import rpyc
import struct

import nengo
import numpy as np

import thread
import time

class View:
    def __init__(self, model, udp_port=56789, client='localhost',
                 default_labels={}, filename=None):
        self.default_labels = default_labels
        self.need_encoders = []
        self.overrides = {}
        self.override_last_time = {}
        self.block_time = 0.0
        self.should_reset = False
        self.should_stop = False

        # connect to the remote java server
        self.rpyc = rpyc.classic.connect(client)

        # make a ValueReceiver on the server to receive UDP data
        # since java leaves a port open for a while, try other ports if
        # the one we specify isn't open
        attempts = 0
        while attempts<1000:
            try:
                vr = self.rpyc.modules.timeview.javaviz.ValueReceiver(
                        udp_port+attempts)
                break
            except:
                attempts += 1
        vr.start()
        self.value_receiver = vr
        self.udp_port = udp_port + attempts


        # for sending packets (with values to be visualized)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP
        self.socket_target = (client, udp_port+attempts)

        # for receiving packets (the slider values from the visualizer)
        self.socket_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_recv.bind(('localhost', udp_port+attempts+1))
        thread.start_new_thread(self.receiver, ())

        # build the dummy model on the server
        self.label = model.label
        if self.label is None:
            self.label = filename
        if self.label is None:
            self.label='Nengo Visualizer 0x%x'%id(model)
        self.label = self.label.replace('.', '_')
        self.label = self.label.replace('/', '_')
        self.label = self.label.replace('\\', '_')
        net = self.rpyc.modules.nef.Network(self.label)

        self.control_node = self.rpyc.modules.timeview.javaviz.ControlNode(
            '(javaviz control)', 'localhost', self.udp_port + 1,
            self.value_receiver)
        net.add(self.control_node)
        self.remote_objs = {}
        self.inputs = []
        self.probe_count = 0
        self.input_count = 0
        self.all_probes = []
        
        self.get_all_probes(model)
        self.process_network(net, model, names=[])
        
        if self.probe_count == 0:
            # need at least one probe to let the synchronizing system work
            # so we make a dummy one
            with model:
                def send(t, self=self):
                    msg = struct.pack('>Lf', 0xFFFFFFFF, t)
                    self.socket.sendto(msg, self.socket_target)
                nengo.Node(send)


        self.model = model
        self.net = net
        self.initialize_vocabs()

    def get_all_probes(self, network):
        for subnet in network.networks:
            self.get_all_probes(subnet)
            self.all_probes.extend(network.probes)

    def initialize_vocabs(self):
        vocabs = []
        if hasattr(self.model, '_modules'):
            for m in self.model._modules.values():
                for obj, v in m.inputs.values():
                    if v not in vocabs:
                        vocabs.append(v)
                for obj, v in m.outputs.values():
                    if v not in vocabs:
                        vocabs.append(v)
        for obj, remote in self.remote_objs.items():
            if hasattr(obj, 'vocab'):
                if obj.vocab not in vocabs:
                    vocabs.append(obj.vocab)
        for local in vocabs:
            remote = self.rpyc.modules.hrr.Vocabulary(
                dimensions=local.dimensions,
                randomize=local.randomize,
                unitary=local.unitary,
                max_similarity=local.max_similarity,
                include_pairs=local.include_pairs)
            for i, k in enumerate(local.keys):
                remote.add(k, self.rpyc.modules.hrr.HRR(
                    data=[float(v) for v in local.vectors[i]]))


    def view(self, config=None):
        try:
            layout = \
                self.rpyc.modules.timeview.view.load_layout_file(self.label, False)

            # check whether has layout == ({}, [], {})
            has_layout = layout is not None and any(layout)
        except:
            has_layout = False

        if config is not None and not has_layout:
            # generate a layout based on the current positions of network nodes in GUI
            view, layout, control = self.generate_layout(self.model, config)
            self.net.set_layout(view, layout, control)

        # open up the visualizer on the server
        view = self.net.view()
        self.control_node.set_view(view)

        # disable components that cannot be used with javaviz
        view.time_control.mode_combobox.enabled = False
        view.time_control.dt_combobox.enabled = False

        if config is not None and not has_layout:
            # destroy the layout we created - unless the user saves their layout,
            # we want to generate a new layout next time javaviz is opened.
            self.rpyc.modules.timeview.view.save_layout_file(self.label, {}, [], {})

        self.remote_view = view

    def get_name(self, names, obj, prefix):
        name = obj.label
        if name == None:
            name = self.default_labels.get(id(obj), None)
            if name is None:
                name = '%s_%d' % (obj.__class__.__name__, id(obj))

            # if the provided name has dots (indicating a hierarchy),
            # ignore them since that'll get filled in by the prefix
            if '.' in name:
                name = name.rsplit('.', 1)[1]

        if prefix != '':
            name = '%s.%s' % (prefix, name)

        counter = 2
        base = name
        while name in names:
            name = '%s (%d)' % (base, counter)
            counter += 1
        names.append(name)
        return name.replace('.', '>')

    def process_network(self, remote_net, network, names, prefix=''):
        semantic = []

        for obj in network.ensembles:
            for probe in self.all_probes:
                if probe.target is obj:
                    name = self.get_name(names, obj, prefix)

                    e = self.rpyc.modules.timeview.javaviz.ProbeNode(
                            self.value_receiver, name)

                    remote_net.add(e)
                    self.remote_objs[obj] = e

                    if obj.dimensions >= 8:
                        semantic.append(obj)
                    break

        for obj in network.nodes:
                name = self.get_name(names, obj, prefix)

                if obj.size_in == 0:
                    if obj.size_out > 0:
                        output = obj.output

                        if callable(output):
                            output = output(0.0)
                        if isinstance(output, (int, float)):
                            output_dims = 1
                        elif isinstance(output, np.ndarray):
                            if output.shape == ():
                                output_dims = 1
                            else:
                                assert len(output.shape) == 1
                                output_dims = output.shape[0]
                        else:
                            output_dims = len(output)
                        obj._output_dims = output_dims
                        input = remote_net.make_input(name, tuple([0]*output_dims))
                        obj.output = OverrideFunction(self, obj.output, 
                                    self.input_count)
                        self.control_node.register(self.input_count, input)
                        self.input_count += 1
                        self.remote_objs[obj] = input
                        self.inputs.append(input)
                else:
                    for probe in self.all_probes:
                        if probe.target is obj:
                            e = self.rpyc.modules.timeview.javaviz.ProbeNode(
                                    self.value_receiver, name)
                            remote_net.add(e)
                            self.remote_objs[obj] = e
                            if obj.size_out >= 8:
                                semantic.append(obj)
                            break

        for subnet in network.networks:
            name = self.get_name(names, subnet, prefix)
            self.process_network(remote_net, subnet, names, prefix=name)

        for c in network.connections:
            # handle direct connections
            pre = c.pre
            if isinstance(pre, nengo.objects.Neurons):
                pre = pre.ensemble
            post = c.post
            if isinstance(post, nengo.objects.Neurons):
                post = post.ensemble

            if pre in self.remote_objs and post in self.remote_objs:
                r_pre = self.remote_objs[pre]
                r_post = self.remote_objs[post]
                if r_pre in self.inputs:
                    oname = 'origin'
                    dims = c.pre._output_dims
                else:
                    oname = 'current'  # a dummy origin
                    dims = 1
                t = r_post.create_new_dummy_termination(dims)
                remote_net.connect(r_pre.getOrigin(oname), t)
            else:
                print 'cannot process connection from %s to %s'%(`c.pre`, `c.post`)

        for obj in semantic:
            # do this after all the nodes and connections so that the
            # nodes and connections we create won't be mapped to remote
            self.add_semantic_override(obj, network)

        for probe in network.probes:

            probe_id = self.probe_count
            self.probe_count += 1

            if isinstance(probe.target, nengo.Ensemble) and probe.attr == 'decoded_output':
                obj = probe.target
                e = self.remote_objs[obj]
                e.add_probe(probe_id, obj.dimensions, 'X')
                with network:
                    def send(t, x, self=self, format='>Lf'+'f'*obj.dimensions,
                            id=probe_id):
                        msg = struct.pack(format, id, t, *x)
                        self.socket.sendto(msg, self.socket_target)

                    node = nengo.Node(send, size_in=obj.dimensions)
                    c = nengo.Connection(obj, node, synapse=None)

            elif isinstance(probe.target, nengo.Ensemble) and probe.attr == 'spikes':
                obj = probe.target
                e = self.remote_objs[obj]
                self.need_encoders.append(obj)
                e.add_spike_probe(probe_id, obj.n_neurons)
                with network:

                    def send(t, x, self=self, id=probe_id):
                        spikes = filter(lambda i: x[i] > 0.5, range(len(x)))
                        num_spikes = len(spikes)
                        format_string = '>LH'+'H'*num_spikes
                        msg = struct.pack(
                             format_string, id, num_spikes, *spikes)
                        self.socket.sendto(msg, self.socket_target)

                    node = nengo.Node(send, size_in=obj.n_neurons)
                    c = nengo.Connection(obj.neurons, node, synapse=None)
            elif isinstance(probe.target, nengo.Node) and probe.attr == 'output':
                obj = probe.target
                e = self.remote_objs[obj]
                if e in self.inputs:
                    # inputs are automatically probed
                    continue
                e.add_probe(probe_id, obj.size_out, 'X')
                with network:
                    def send(t, x, self=self, format='>Lf'+'f'*obj.size_out,
                            id=probe_id):
                        msg = struct.pack(format, id, t, *x)
                        self.socket.sendto(msg, self.socket_target)

                    node = nengo.Node(send, size_in=obj.size_out)
                    c = nengo.Connection(obj, node, synapse=None)

            else:
                print 'Unhandled probe', probe

    def add_semantic_override(self, obj, network):
        if isinstance(obj, nengo.Ensemble):
            dim = obj.dimensions
        elif isinstance(obj, nengo.Node):
            dim = obj.size_out
        remote = self.remote_objs[obj]
        self.control_node.register(self.input_count, remote)
        override_func = PassthroughOverrideFunction(self, self.input_count)
        with network:
            node = nengo.Node(override_func, size_in=dim, size_out=dim)
            nengo.Connection(obj, node, synapse=None)
            nengo.Connection(node, obj, synapse=0.05)

        self.input_count += 1




    def update_model(self, sim):
        """Grab data from the simulator needed for plotting."""

        for obj in self.need_encoders:
            remote = self.remote_objs[obj]
            encoders = sim.model.params[obj].encoders
            remote.set_encoders(obj.n_neurons, obj.dimensions,
                    tuple([float(x) for x in encoders.flatten()]))


    def receiver(self):
        # watch for packets coming from the server.  There should be one
        # packet every time step, with the current time and any slider values
        # that are set
        self.socket_recv.settimeout(0.2)
        while True:
            try:
                msg = self.socket_recv.recv(4096)
            except socket.timeout:
                if self.value_receiver.should_close:
                    break
                else:
                    continue
            # grab the current time the simulator thinks it is at
            time = struct.unpack('>f', msg[:4])

            if self.block_time > 0 and time[0] < self.block_time:
                self.should_reset = True

            # tell the simulator to stop if it is past the given time
            self.block_time = time[0]

            # override the node output values if the visualizer says to
            for i in range((len(msg)-4)/12):
                id, index, value = struct.unpack('>LLf', msg[4+i*12:16+i*12])
                self.overrides[id][index] = value
                self.override_last_time[id] = time[0]
        self.socket.close()
        self.socket_recv.close()
        self.should_stop = True

    def generate_layout(self, network, config):
        window_width = 1000
        window_height = 625
        window_pos_x = 50
        window_pos_y = 50

        top_cushion = int(0.1 * window_height)
        bottom_cushion = 200
        left_cushion = int(0.1 * window_width)
        right_cushion = 200

        draw_area_height = window_height - top_cushion - bottom_cushion
        draw_area_width = window_width - left_cushion - right_cushion

        view = {'state':0, 'height':window_height, 'width':window_width,
                'x':window_pos_x, 'y':window_pos_y}

        def _generate_layout(layout, network, config):
            # populate the list 'layout' with an item specifying
            # the location of each node/ensemble
            for obj in network.nodes + network.ensembles:
                if obj in self.remote_objs:
                    name = self.remote_objs[obj].getName()
                    pos = config[obj].pos
                    layout_item = (name, None,
                           {'label': False, 'x': int(pos[0]), 'y': int(pos[1]),
                            'width': 100, 'height': 20})
                    layout.append(layout_item)

            for obj in network.networks:
                _generate_layout(layout, obj, config)

        layout = []
        _generate_layout(layout, network, config)

        # transform the node locations so the whole network is visible in javaviz
        points = [[item[2]['x'], item[2]['y']] for item in layout]
        points = np.array(points)

        maxes = np.max(points, 0)
        mins = np.min(points, 0)

        points -= mins + (maxes - mins) / 2.0
        points = (points * np.array([draw_area_width, draw_area_height]) )/ (maxes - mins)
        points += np.array([left_cushion + draw_area_width / 2.0,
                            top_cushion + draw_area_height / 2.0])
        points = points.astype('int')

        for point, item in zip(points, layout):
            item[2]['x'] = point[0]
            item[2]['y'] = point[1]

        control = {}

        return view, layout, control


# this replaces any callable nengo.Node's function with a function that:
# a) blocks if it gets ahead of the time the visualizer wants to show
# b) uses the slider value sent back from the visualizer instead of the
# output function, if that slider has been set
class OverrideFunction(object):
    def __init__(self, view, function, id):
        self.view = view
        self.function = function
        self.id = id
        self.view.overrides[id] = {}
    def check_reset(self):
        if self.view.should_reset:
            for k in self.view.overrides.keys():
                self.view.overrides[k] = {}
            self.view.override_last_time.clear()
            self.view.should_reset = False
            self.view.remote_view.sim_time = 0.0

            raise VisualizerResetException()

    def __call__(self, t):
        self.check_reset()
        while self.view.block_time < t:
            time.sleep(0.01)
            if self.view.should_stop:
                raise VisualizerExitException('JavaViz closed')
            self.check_reset()
        if callable(self.function):
            value = np.array(self.function(t), dtype='float')
        else:
            value = np.array(self.function, dtype='float')
        if len(value.shape) == 0:
            value.shape = (1,)
        for k,v in self.view.overrides.get(self.id, {}).items():
            value[k] = v
        return value

# This is meant for driving a Nengo node towards a particular value.
# The idea is to have a Node that takes input from a nengo Object and sends
# output back to that same object.  The "override" value comes from the
# javaviz visualizer.  If there is no override value (or if there hasn't
# been one for the last few time steps), then this node outputs a 0.
# Otherwise, it outputs the difference between the value it is receiving
# and the target value.  This should drive the nengo object it is connected
# to towards the override value.
class PassthroughOverrideFunction(object):
    def __init__(self, view, id):
        self.view = view
        self.id = id
        self.view.overrides[id] = {}
    def check_reset(self):
        if self.view.should_reset:
            for k in self.view.overrides.keys():
                self.view.overrides[k] = {}
            self.view.override_last_time.clear()
            self.view.should_reset = False
            self.view.remote_view.sim_time = 0.0

            raise VisualizerResetException()

    def __call__(self, t, x):
        self.check_reset()
        while self.view.block_time < t:
            time.sleep(0.01)
            if self.view.should_stop:
                raise VisualizerExitException('JavaViz closed')
            self.check_reset()
        value = np.array(x)
        last_time = self.view.override_last_time.get(self.id, None)
        if last_time is not None and t - last_time < 0.002:
            for k,v in self.view.overrides.get(self.id, {}).items():
                value[k] = v
        return 10*(value - x)


class VisualizerExitException(Exception):
    pass

class VisualizerResetException(Exception):
    pass


