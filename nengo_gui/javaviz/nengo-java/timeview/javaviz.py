import nef

import struct

class Ensemble(nef.Node):
    def __init__(self, receiver, id,  name, dimensions):
        nef.Node.__init__(self, name)
        receiver.register(id, self)
        self.termination_count = 0
        self.output = self.make_output('X', dimensions)
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
        self.ensembles = {}

    def register(self, id, ensemble):
        self.ensembles[id] = ensemble

    def run(self):
        while True:
            self.socket.receive(self.packet)

            d = java.io.DataInputStream(java.io.ByteArrayInputStream(self.packet.getData()))

            id = d.readInt()

            ensemble = self.ensembles[id]
            time = d.readFloat()
            length = len(ensemble.output._value)
            for i in range(length):
                ensemble.output._value[i] = d.readFloat()

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


