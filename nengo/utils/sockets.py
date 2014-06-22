import struct
import socket
# import nengo
import Queue
import time
import threading
import errno


class SocketCheckAliveThread(threading.Thread):
    def __init__(self, socket_class):
        threading.Thread.__init__(self)
        self.socket = socket_class
        self.name = str(int(time.time()))

    def run(self):
        # Keep checking if the socket class is still being used.
        while (time.time() - self.socket.last_active <
               self.socket.max_idle_time_current):
            time.sleep(self.socket.alive_thread_sleep_time)
        # If the socket class is idle, terminate the socket
        self.socket._close_socket()


class UDPSocket:
    def __init__(self, dimensions=1, dt=0.001, local_port=-1,
                 dest_addr='127.0.0.1', dest_port=-1,
                 timeout=30, max_idle_time=1):
        self.dim = dimensions
        self.local_addr = '127.0.0.1'
        self.local_port = local_port
        self.dest_addr = dest_addr
        self.dest_port = dest_port
        self.timeout = timeout
        self.dt = dt
        self.byte_order = '!'

        self.last_active = time.time()
        self.max_idle_time_initial = max_idle_time
        self.max_idle_time_timeout = max(max_idle_time, timeout + 1)
        self.max_idle_time_current = self.max_idle_time_timeout
        self.alive_check_thread = None
        self.alive_thread_sleep_time = max_idle_time / 2.0
        self.retry_backoff_time = 1

        self.socket = None
        self.is_sender = dest_port > 0
        self.is_receiver = local_port > 0
        self.ignore_timestamp = False

        self.max_len = (self.dim + 1) * 4
        self.value = [0] * self.dim
        self.buffer = Queue.PriorityQueue()

    def __del__(self):
        self.close()

    def _initialize(self):
        self.value = 0
        self.timeout_count = 0

        while (not self.buffer.empty()):
            self.buffer.get()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.local_addr, max(self.local_port, 0)))
            self.socket.settimeout(self.timeout)
        except socket.error:
            raise RuntimeError("UDPSocket: Could not bind to socket. "
                               "Address: %s, Port: %s, is in use. If "
                               "simulation has been run before, wait for "
                               "previous UDPSocket to release the port. "
                               "(See 'max_idle_time' argument in class "
                               "constructor, currently set to %f seconds)" %
                               (self.local_addr, self.local_port,
                                self.max_idle_time_current))

        self.last_active = time.time()
        self.alive_check_thread = SocketCheckAliveThread(self)
        self.alive_check_thread.start()

    def _config_wipe(self):
        self.local_addr = '127.0.0.1'
        self.local_port = -1
        self.dest_addr = '127.0.0.1'
        self.dest_port = -1
        self.is_sender = False
        self.is_receiver = False
        self.ignore_timestamp = False

        self.close()

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
        # If t == 0, return array of zeros. Terminate any open sockets to
        # reset system
        if (t == 0):
            self.close()
            return [0] * self.dim
        # Initialize socket if t > 0, and it has not been initialized
        if (t > 0 and self.socket is None):
            self._initialize()

        self.last_active = time.time()

        if (self.is_sender):
            self.socket.sendto(self.pack_packet(t, x),
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
                elif (t_peek >= t + self.dt):
                    # Timestamp of first item in buffer is > t+dt (i.e. all
                    # items in the buffer are future packets). Assume packet
                    # for current timestep has been lost.
                    found_item = True

            while (not found_item):
                try:
                    packet, addr = self.socket.recvfrom(self.max_len)
                    t_data, value = self.unpack_packet(packet)
                    if (t_data >= t and t_data < t + self.dt) or \
                       self.ignore_timestamp:
                        self.value = value
                        found_item = True
                    elif (t_data >= t + self.dt):
                        self.buffer.put((t_data, value))
                        found_item = True

                    # Packet recv success! Decay max_idle_time_current time
                    # to the user specified max_idle_time (which can be
                    # smaller than the socket timeout time)
                    self.max_idle_time_current = \
                        max(self.max_idle_time_initial,
                            self.max_idle_time_current * 0.9)
                    self.retry_backoff_time = \
                        max(1, self.retry_backoff_time / 2)

                except (socket.error, AttributeError) as error:
                    # Socket error has occured. Probably a timeout.
                    # Assume worst case, set max_idle_time_current to
                    # max_idle_time_timeout to wait for more timeouts to
                    # occur (this is so that the socket isn't constantly
                    # closed by the check_alive thread)
                    self.max_idle_time_current = self.max_idle_time_timeout

                    # Timeout occured, assume packet lost.
                    found_item = True

                    # If connection was reset (somehow?), close the socket.
                    # In this case, retry the connection, and retry receiving
                    # the packet again.
                    if (error.errno == errno.ECONNRESET):
                        self._retry_connection()
                        found_item = False

        return self.value

    def _terminate_alive_check_thread(self):
        self.max_idle_time_current = self.max_idle_time_timeout
        if (not self.alive_check_thread is None):
            self.last_active = 0
            self.alive_check_thread.join()
        self.alive_check_thread = None

    def _close_socket(self):
        if (not self.socket is None):
            self.socket.close()
        self.socket = None

    def _retry_connection(self):
        self.socket = None
        while (self.socket is None):
            time.sleep(self.retry_backoff_time)
            try:
                self.close()
                self._initialize()
            except socket.error:
                pass
            self.retry_backoff_time *= 2

    def close(self):
        self._terminate_alive_check_thread()
        self._close_socket()
