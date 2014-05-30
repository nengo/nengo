import socket
import rpyc
import struct

import nengo
import numpy as np

import thread
import time

class View:
    def __init__(self, model, udp_port=56789, client='localhost'):
        # connect to the remote java server
        self.rpyc = rpyc.classic.connect(client)

        # make a ValueReceiver on the server to receive UDP data
        # since java leaves a port open for a while, try other ports if
        # the one we specify isn't open
        attempts = 0
        while attempts<100:
            try:
                vr = self.rpyc.modules.timeview.javaviz.ValueReceiver(udp_port+attempts)
                break
            except:
                attempts += 1
        vr.start()


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

        control_ensemble = None
        remote_objs = {}
        ignore_connections = set()
        inputs = []
        for obj in model.ensembles:
                if control_ensemble is None:
                    e = self.rpyc.modules.timeview.javaviz.ControlEnsemble(vr, id(obj)&0xFFFF, obj)
                    e.init_control('localhost', udp_port+attempts+1)
                    control_ensemble = e
                else:
                    e = self.rpyc.modules.timeview.javaviz.Ensemble(vr, id(obj)&0xFFFF, obj)
                net.add(e)
                remote_objs[obj] = e


                with model:
                    def send(t, x, self=self, format='>Lf'+'f'*obj.dimensions, id=id(obj)&0xFFFF):
                        msg = struct.pack(format, id, t, *x)
                        self.socket.sendto(msg, self.socket_target)


                    node = nengo.Node(send, size_in=obj.dimensions)
                    c = nengo.Connection(obj, node, synapse=None)
                    ignore_connections.add(c)
        for obj in model.nodes:
                if obj.size_in == 0:
                    output = obj.output

                    if callable(output):
                        output = output(0.0)#np.zeros(obj.size_in))
                    if isinstance(output, (int, float)):
                        output_dims = 1
                    else:
                        output_dims = len(output)
                    obj._output_dims = output_dims
                    input = net.make_input(obj.label, tuple([0]*output_dims))
                    obj.output = OverrideFunction(obj.output, id(input)&0xFFFF)
                    remote_objs[obj] = input
                    inputs.append(input)

        for input in inputs:
            control_ensemble.register(id(input)&0xFFFF, input)

        for c in model.connections:
            if c not in ignore_connections:
                if c.pre in remote_objs and c.post in remote_objs:
                    pre = remote_objs[c.pre]
                    post = remote_objs[c.post]
                    dims = c.pre.dimensions if isinstance(c.pre, nengo.Ensemble) else c.pre._output_dims
                    t = post.create_new_dummy_termination(dims)
                    net.connect(pre, t)
                else:
                    print 'cannot process connection from %s to %s'%(`c.pre`, `c.post`)

        # open up the visualizer on the server
        view = net.view()
        control_ensemble.set_view(view)

    def receiver(self):
        # watch for packets coming from the server.  There should be one
        # packet every time step, with the current time and any slider values
        # that are set
        print 'waiting for msg'
        while True:
            msg = self.socket_recv.recv(4096)
            # grab the current time the simulator thinks it is at
            time = struct.unpack('>f', msg[:4])
            # tell the simulator to stop if it is past the given time
            OverrideFunction.overrides['block_time'] = time[0]

            # override the node output values if the visualizer says to
            for i in range((len(msg)-4)/12):
                id, index, value = struct.unpack('>LLf', msg[4+i*12:16+i*12])
                OverrideFunction.overrides[id][index]=value

# this replaces any callable nengo.Node's function with a function that:
# a) blocks if it gets ahead of the time the visualizer wants to show
# b) uses the slider value sent back from the visualizer instead of the
# output function, if that slider has been set
class OverrideFunction(object):
    overrides = {'block_time':0.0}
    def __init__(self, function, id):
        self.function = function
        self.id = id
        OverrideFunction.overrides[id] = {}
    def __call__(self, t):
        while OverrideFunction.overrides['block_time'] < t:
            time.sleep(0.01)
        if callable(self.function):
            value = np.array(self.function(t))
        else:
            value = np.array(self.function)
        for k,v in OverrideFunction.overrides.get(self.id, {}).items():
            value[k] = v
        return value
