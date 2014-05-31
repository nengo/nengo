import nef

import struct

class ProbeNode(nef.Node):
    def __init__(self, receiver, name):
        nef.Node.__init__(self, name)
        self.termination_count = 0
        self.probes = {}
        self.receiver = receiver

        # make a dummy connection for drawing arrow in the gui.
        #  we use "current" since that is ignored by the
        #  interactive plots visualizer
        self.make_output('current', 1)

    def add_probe(self, id, dimensions, origin_name):
        self.probes[id] = self.make_output(origin_name, dimensions)
        self.receiver.register(id, self.probes[id])

    def create_new_dummy_termination(self, dimensions):
        name = 'term%d'%self.termination_count
        self.make_input(name, dimensions)
        self.termination_count += 1
        return self.getTermination(name)


import java
import jarray
class ValueReceiver(java.lang.Thread):
    def __init__(self, port):
        self.socket = java.net.DatagramSocket(port)
        maxLength = 65535
        self.buffer = jarray.zeros(maxLength,'b')
        self.packet = java.net.DatagramPacket(self.buffer, maxLength)
        self.probes = {}

    def register(self, id, probe):
        self.probes[id] = probe

    def run(self):
        while True:
            self.socket.receive(self.packet)

            d = java.io.DataInputStream(java.io.ByteArrayInputStream(self.packet.getData()))

            id = d.readInt()

            probe = self.probes[id]
            time = d.readFloat()
            length = len(probe._value)
            for i in range(length):
                probe._value[i] = d.readFloat()

class ControlNode(nef.Node):
    def __init__(self, name, address, port, dt=0.001):
        nef.Node.__init__(self, name)
        self.view = None
        self.socket = java.net.DatagramSocket()
        self.address = java.net.InetAddress.getByName(address)
        self.port = port
        self.inputs = {}
        self.dt = dt
        self.formats = {}
        self.ids = {}
    def set_view(self, view):
        self.view = view
    def register(self, id, input):
        self.inputs[id] = input
        self.formats[input] = '>Lf'+'f'*input.getOrigin('origin').getDimensions()
        self.ids[input.name] = id
    def start(self):
        cache = {}

        while True:
            if self.view is not None:
                msg = struct.pack('>f', self.t)
                for key, value in self.view.forced_origins.items():
                    (name, origin, index) = key
                    if origin=='origin':
                        id = self.ids.get(name, None)
                        if id is not None:
                            prev = cache.get((id,index),None)
                            if value != prev:
                                msg += struct.pack('>LLf', id, index, value)
                                cache[(id, index)] = value
                packet = java.net.DatagramPacket(msg, len(msg), self.address, self.port)
                self.socket.send(packet)

            yield self.dt


