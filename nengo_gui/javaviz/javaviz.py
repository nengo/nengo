import socket
import rpyc
import struct

import nengo
import numpy as np

import thread
import time

class View:
    def __init__(self, model, udp_port=56789, client='localhost',
                 default_labels={}):
        self.default_labels = default_labels
        self.overrides = {}
        self.block_time = 0.0
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
        print 'Using port', self.udp_port


        # for sending packets (with values to be visualized)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #UDP
        self.socket_target = (client, udp_port+attempts)

        # for receiving packets (the slider values from the visualizer)
        self.socket_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_recv.bind(('localhost', udp_port+attempts+1))
        thread.start_new_thread(self.receiver, ())

        # build the dummy model on the server
        label = model.label
        if label is None: label='Nengo Visualizer 0x%x'%id(model)
        net = self.rpyc.modules.nef.Network(label)

        self.control_node = self.rpyc.modules.timeview.javaviz.ControlNode(
            '(javaviz control)', 'localhost', self.udp_port + 1,
            self.value_receiver)
        net.add(self.control_node)
        self.remote_objs = {}
        self.inputs = []
        self.probe_count = 0

        self.process_network(net, model, names=[])

        for input in self.inputs:
            self.control_node.register(id(input)&0xFFFF, input)

        # open up the visualizer on the server
        view = net.view()
        self.control_node.set_view(view)


    def get_name(self, names, obj, prefix):
        name = obj.label
        if (isinstance(obj, nengo.Ensemble) and name == 'Ensemble' or
                isinstance(obj, nengo.Node) and name == 'Node' or
                isinstance(obj, nengo.Network) and name == None):
            name = self.default_labels.get(id(obj), name)

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
        return name

    def process_network(self, remote_net, network, names, prefix=''):
        for obj in network.ensembles:
            name = self.get_name(names, obj, prefix)

            e = self.rpyc.modules.timeview.javaviz.ProbeNode(
                    self.value_receiver, name)

            remote_net.add(e)
            self.remote_objs[obj] = e

        for obj in network.nodes:
            name = self.get_name(names, obj, prefix)
            if obj.size_in == 0:
                output = obj.output

                if callable(output):
                    output = output(0.0)#np.zeros(obj.size_in))
                if isinstance(output, (int, float)):
                    output_dims = 1
                else:
                    output_dims = len(output)
                obj._output_dims = output_dims
                input = remote_net.make_input(name, tuple([0]*output_dims))
                obj.output = OverrideFunction(self, obj.output, id(input)&0xFFFF)
                self.remote_objs[obj] = input
                self.inputs.append(input)
            else:
                e = self.rpyc.modules.timeview.javaviz.ProbeNode(
                        self.value_receiver, name)
                remote_net.add(e)
                self.remote_objs[obj] = e

        for subnet in network.networks:
            name = self.get_name(names, subnet, prefix)
            self.process_network(remote_net, subnet, names, prefix=name)

        for c in network.connections:
            if c.pre in self.remote_objs and c.post in self.remote_objs:
                pre = self.remote_objs[c.pre]
                post = self.remote_objs[c.post]
                if pre in self.inputs:
                    oname = 'origin'
                    dims = c.pre._output_dims
                else:
                    oname = 'current'  # a dummy origin
                    dims = 1
                t = post.create_new_dummy_termination(dims)
                remote_net.connect(pre.getOrigin(oname), t)
            else:
                print 'cannot process connection from %s to %s'%(`c.pre`, `c.post`)

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
                e.add_spike_probe(probe_id, obj.n_neurons)
                with network:
                    def send(t, x, self=self, format='>Lf'+'f'*obj.n_neurons,
                            id=probe_id):
                        msg = struct.pack(format, id, t, *x)
                        self.socket.sendto(msg, self.socket_target)

                    node = nengo.Node(send, size_in=obj.n_neurons)
                    c = nengo.Connection(obj.neurons, node, synapse=None)
            else:
                print 'Unhandled probe', probe


    def receiver(self):
        # watch for packets coming from the server.  There should be one
        # packet every time step, with the current time and any slider values
        # that are set
        print 'waiting for msg'
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
            # tell the simulator to stop if it is past the given time
            self.block_time = time[0]

            # override the node output values if the visualizer says to
            for i in range((len(msg)-4)/12):
                id, index, value = struct.unpack('>LLf', msg[4+i*12:16+i*12])
                self.overrides[id][index]=value
        self.socket.close()
        print 'finished receiving'

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
    def __call__(self, t):
        while self.view.block_time < t:
            time.sleep(0.01)
        if callable(self.function):
            value = np.array(self.function(t), dtype='float')
        else:
            value = np.array(self.function, dtype='float')
        for k,v in self.view.overrides.get(self.id, {}).items():
            value[k] = v
        return value
