import struct
import socket
# import nengo
import Queue
import time
import thread


class UDPSocket:
    def __init__(self, dim=1, dt=0.001, local_port=-1,
                 dest_addr='127.0.0.1', dest_port=-1,
                 timeout=5, max_idle_time=60):
        self.dim = dim
        self.local_addr = '127.0.0.1'
        self.local_port = local_port
        self.dest_addr = dest_addr
        self.dest_port = dest_port
        self.timeout = timeout
        self.dt = dt
        self.byte_order = '!'

        self.last_active = time.time()
        self.max_idle_time = max_idle_time

        self.socket = None
        self.is_sender = False
        self.is_receiver = False
        self.ignore_timestamp = False

        self.max_len = (dim + 1) * 4
        self.value = [0] * dim
        self.buffer = Queue.PriorityQueue()

    def _initialize(self):
        self.value = 0
        self.timeout_count = 0

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.local_addr, max(self.local_port, 0)))
        self.socket.settimeout(self.timeout)

        self.last_active = time.time()
        thread.start_new_thread(self._alive_check, self)

    def _config_wipe(self):
        self.local_addr = '127.0.0.1'
        self.local_port = -1
        self.dest_addr = '127.0.0.1'
        self.dest_port = -1
        self.is_sender = False
        self.is_receiver = False
        self.ignore_timestamp = False

        if not self.socket is None:
            self.close()

    def _alive_check(self):
        while (time.time() - self.last_active < self.max_idle_time):
            time.sleep(1)
        self._close()

    def config_send_only(self, dest_addr, dest_port):
        self._config_wipe()
        if (dest_port > 0):
            self.is_sender = True
            self.dest_addr = dest_addr
            self.dest_port = dest_port
        else:
            raise ValueError("UDPSocket: Invalid send only configuration."
                             "Destination port should be > 0")

    def config_recv_only(self, local_port, timeout=5, ignore_timestamp=False):
        self._config_wipe()
        if (local_port > 0):
            self.is_receiver = True
            self.local_port = local_port
            self.timeout = timeout
            self.ignore_timestamp = ignore_timestamp
        else:
            raise ValueError("UDPSocket: Invalid recv only configuration."
                             "Local port should be > 0")

    def config_send_recv(self, local_port, dest_addr, dest_port, timeout=5,
                         ignore_timestamp=False):
        self._config_wipe()
        if (local_port > 0 and dest_port > 0):
            self.is_sender = True
            self.is_receiver = True
            self.dest_addr = dest_addr
            self.dest_port = dest_port
            self.local_port = local_port
            self.timeout = timeout
            self.ignore_timestamp = ignore_timestamp
        else:
            raise ValueError("UDPSocket: Invalid send and recv configuration."
                             "Both destination and local ports should be > 0")

    def set_byte_order(self, byte_order):
        if (byte_order.lower() == "little"):
            self.byte_order = '<'
        elif (byte_order.lower() == "big"):
            self.byte_order = '>'
        else:
            self.byte_order = byte_order

    def pack_packet(self, t, x):
        # pack_packet takes a timestamp and data (x) and makes a socket packet
        # Default packet data type: float
        # Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]
        send_data = [float(t + self.dt / 2.0)] + \
                    [x[i] for i in range(self.dim)]
        packet = struct.pack(self.byte_order + 'f' * (self.dim + 1),
                             *send_data)
        return packet

    def unpack_packet(self, packet):
        # unpack_packet takes a packet and extracts a timestamp and data (x)
        # Default packet data type: float
        # Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]
        data_len = len(packet) / 4
        data = list(struct.unpack(self.byte_order + 'f' * data_len, packet))
        t_data = data[0]
        value = data[1:]
        return t_data, value

    def run(self, t, x=None):
        self.last_active = time.time()

        # Initialize socket if t > 0, and it has not been initialized
        if (t > 0 and self.socket is None):
            self._initialize()
        if (t == 0):
            return self.value

        # If the number of timeouts has exceeded the maximum number of timeouts
        # auto-close the socket
        if (self.timeout_count > self.timeouts_to_close):
            self._close()

        if (self.is_sender):
            self.socket.sendto(self.make_packet(t, x),
                               (self.dest_addr, self.dest_port))
        if (self.is_receiver):
            found_item = False
            if (not self.buffer.empty()):
                # There are items (packets with future timestamps) in the
                # buffer. Therefore, check the buffer for appropriate
                # information
                t_peek = self.buffer.queue[0][0]
                if (t_peek >= t and t_peek < t + self.dt) or \
                   self.ignore_timestamp:
                    # Timestamp of first item in buffer is > t && < t+dt,
                    # meaning that this is the information for the current
                    # timestep, so it should be used.
                    data = self.buffer.get()
                    self.value = data[1]
                    found_item = True
                elif(t_peek >= t + self.dt):
                    # Timestamp of first item in buffer is > t+dt (i.e. all
                    # items in the buffer are future packets). Assume packet
                    # for current timestep has been lost.
                    found_item = True
            while(not found_item):
                try:
                    packet, addr = self.socket.recvfrom(self.max_len)
                    t_data, value = self.unpack_packet(packet)
                    if (t_data >= t and t_data < t + self.dt) or \
                       self.ignore_timestamp:
                        self.value = value
                        found_item = True
                    elif (t_data >= t):
                        self.buffer.put((t_data, value))
                        found_item = True
                except:
                    # Socket error has occured. Probably a timeout.
                    found_item = True
        return self.value

    def close(self):
        self.socket.close()
        self.socket = None

# -------------------------------
# import matplotlib.pyplot as plt

# dt = 0.001

# udp_local_addr = "127.0.0.1"
# udp_local_port = 12346
# udp_dest_addr = "127.0.0.1"
# udp_dest_port = 12345

# UDP = UDPSocket(dim=1, dt=dt)
# #UDP.config_send_recv(local_port=udp_local_port, dest_addr=udp_dest_addr,
# #                     dest_port=udp_dest_port)
# #UDP.config_send_only(dest_addr=udp_dest_addr, dest_port=udp_dest_port)
# UDP.config_recv_only(local_port=udp_local_port)
# m = nengo.Model("test")
# input = m.make_node("Input", lambda t: UDP.run(t))
# m.probe("Input", filter=0)

# s = m.simulator(dt=dt)
# #s.run(0.002)
# s.run(1)
# UDP.close()

# t = s.data(m.t)

# plt.plot(t, s.data("Input"), label="ref")
# plt.show()
