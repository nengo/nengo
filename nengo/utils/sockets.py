from __future__ import absolute_import

import struct
import socket
import time
import threading
import errno
import warnings

from nengo.utils.compat import queue


class SocketCheckAliveThread(threading.Thread):
    def __init__(self, socket_class):
        threading.Thread.__init__(self)
        self.socket_class = socket_class

    def run(self):
        # Keep checking if the socket class is still being used.
        while (time.time() - self.socket_class.last_active <
               self.socket_class.max_idle_time_current):
            time.sleep(self.socket_class.alive_thread_sleep_time)
        # If the socket class is idle, terminate the sockets
        self.socket_class._close_recv_socket()
        self.socket_class._close_send_socket()


class UDPSocket(object):
    def __init__(self, send_dim=1, recv_dim=1, dt_remote=0,
                 local_port=-1, dest_addr='127.0.0.1', dest_port=-1,
                 timeout=30, max_idle_time=1):
        self.local_addr = '127.0.0.1'
        self.local_port = local_port
        self.dest_addr = (dest_addr if isinstance(dest_addr, list)
                          else [dest_addr])
        self.dest_port = (dest_port if isinstance(dest_port, list)
                          else [dest_port])
        self.timeout = timeout
        self.byte_order = '!'

        self.last_t = 0.0
        self.dt = 0.0                               # Local simulation dt
        self.dt_remote = max(dt_remote, self.dt)    # dt btw each packet sent
        self.last_packet_t = 0.0

        self.last_active = time.time()
        self.max_idle_time_initial = max_idle_time
        self.max_idle_time_timeout = max(max_idle_time, timeout + 1)
        self.max_idle_time_current = self.max_idle_time_timeout
        self.alive_check_thread = None
        self.alive_thread_sleep_time = max_idle_time / 2.0
        self.retry_backoff_time = 1

        self.send_socket = None
        self.recv_socket = None
        self.is_sender = dest_port != -1
        self.is_receiver = local_port != -1
        self.ignore_timestamp = False

        self.send_dim = send_dim
        self.recv_dim = recv_dim

        self.max_recv_len = (recv_dim + 1) * 4
        self.value = [0.0] * recv_dim
        self.buffer = queue.PriorityQueue()

    def __del__(self):
        self.close()

    def _initialize(self):
        self.value = [0.0] * self.recv_dim
        self.last_t = 0.0
        self.last_packet_t = 0.0

        # Empty the buffer
        while not self.buffer.empty():
            self.buffer.get()

    def _open_socket(self):
        # Close socket, terminate alive check thread
        self.close()

        if self.is_sender:
            try:
                self.send_socket = socket.socket(socket.AF_INET,
                                                 socket.SOCK_DGRAM)
                self.send_socket.bind((self.local_addr, 0))
            except socket.error as error:
                raise RuntimeError("UDPSocket: Error str: " + str(error))

        if self.is_receiver:
            self._open_recv_socket()

        self.last_active = time.time()
        self.alive_check_thread = SocketCheckAliveThread(self)
        self.alive_check_thread.start()

    def _open_recv_socket(self):
        try:
            self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_socket.bind((self.local_addr, self.local_port))
            self.recv_socket.settimeout(self.timeout)
        except socket.error:
            raise RuntimeError("UDPSocket: Could not bind to socket. "
                               "Address: %s, Port: %s, is in use. If "
                               "simulation has been run before, wait for "
                               "previous UDPSocket to release the port. "
                               "(See 'max_idle_time' argument in class "
                               "constructor, currently set to %f seconds)" %
                               (self.local_addr, self.local_port,
                                self.max_idle_time_current))

    def _retry_connection(self):
        self._close_recv_socket()
        while self.recv_socket is None:
            time.sleep(self.retry_backoff_time)
            try:
                self._open_recv_socket()
            except socket.error:
                pass
            # Failed to open receiving socket, double backoff time, then retry
            self.retry_backoff_time *= 2

    def _terminate_alive_check_thread(self):
        self.max_idle_time_current = self.max_idle_time_timeout
        if self.alive_check_thread is not None:
            self.last_active = 0
            self.alive_check_thread.join()
        self.alive_check_thread = None

    def _close_send_socket(self):
        if self.send_socket is not None:
            self.send_socket.close()
        self.send_socket = None

    def _close_recv_socket(self):
        if self.recv_socket is not None:
            self.recv_socket.close()
        self.recv_socket = None

    def _config_wipe(self):
        self.local_addr = '127.0.0.1'
        self.local_port = -1
        self.dest_addr = '127.0.0.1'
        self.dest_port = -1
        self.is_sender = False
        self.is_receiver = False
        self.ignore_timestamp = False

        self.close()
        self._initialize()

    def config_send_only(self, dest_addr, dest_port):
        self._config_wipe()
        if dest_port > 0:
            self.is_sender = True
            self.dest_addr = dest_addr
            self.dest_port = dest_port
        else:
            raise ValueError("UDPSocket: Invalid send only configuration."
                             "Destination port should be > 0")

    def config_recv_only(self, local_port, timeout=5, ignore_timestamp=False):
        self._config_wipe()
        if local_port > 0:
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
        if local_port > 0 and dest_port > 0:
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
        if byte_order.lower() == "little":
            self.byte_order = '<'
        elif byte_order.lower() == "big":
            self.byte_order = '>'
        else:
            self.byte_order = byte_order

    def pack_packet(self, t, x):
        """Takes a timestamp and data (x) and makes a socket packet

        Default packet data type: float
        Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]"""

        send_data = [float(t + self.dt_remote / 2.0)] + \
                    [x[i] for i in range(self.send_dim)]
        packet = struct.pack(self.byte_order + 'f' * (self.send_dim + 1),
                             *send_data)
        return packet

    def unpack_packet(self, packet):
        """Takes a packet and extracts a timestamp and data (x)

        Default packet data type: float
        Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]"""

        data_len = len(packet) / 4
        data = list(struct.unpack(self.byte_order + 'f' * data_len, packet))
        t_data = data[0]
        value = data[1:]
        return t_data, value

    def __call__(self, t, x=None):
        return self.run(t, x)

    # TODO: name this something better
    def _t_check(t_lim, t):
        return (t_lim >= t and t_lim < t + self.dt) or self.ignore_timestamp:

    def run(self, t, x=None):  # noqa: C901
        # If t == 0, return array of zeros. Terminate any open sockets to
        # reset system
        if t == 0:
            self._initialize()
            self.close()
            return self.value
        # Initialize socket if t > 0, and it has not been initialized
        if t > 0 and ((self.recv_socket is None and self.is_receiver) or
           (self.send_socket is None and self.is_sender)):
            self._open_socket()

        # Calculate dt
        self.dt = t - self.last_t
        self.dt_remote = max(self.dt_remote, self.dt)
        self.last_t = t * 1.0
        self.last_active = time.time()

        if self.is_sender:
            # Calculate if it is time to send the next packet.
            # Time to send next packet if time between last packet and current
            # t + half of dt is >= remote dt
            if (t - self.last_packet_t + self.dt / 2.0 >= self.dt_remote):
                for addr in self.dest_addr:
                    for port in self.dest_port:
                        self.send_socket.sendto(self.pack_packet(t * 1.0, x),
                                                (addr, port))
                self.last_packet_t = t * 1.0   # Copy t (which is an np.scalar)

        if self.is_receiver:
            found_item = False
            if not self.buffer.empty():
                # There are items (packets with future timestamps) in the
                # buffer. Therefore, check the buffer for appropriate
                # information
                t_peek = self.buffer.queue[0][0]
                if _t_check(t_peek, t):
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

            while not found_item:
                try:
                    packet, addr = self.recv_socket.recvfrom(self.max_recv_len)
                    t_data, value = self.unpack_packet(packet)
                    if _t_check(t_data, t):
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
                    found_item = False

                    # Socket error has occured. Probably a timeout.
                    # Assume worst case, set max_idle_time_current to
                    # max_idle_time_timeout to wait for more timeouts to
                    # occur (this is so that the socket isn't constantly
                    # closed by the check_alive thread)
                    self.max_idle_time_current = self.max_idle_time_timeout

                    # Timeout occured, assume packet lost.
                    if isinstance(error, socket.timeout):
                        found_item = True

                    # If connection was reset (somehow?), or closed by the
                    # idle timer (prematurely).
                    # In this case, retry the connection, and retry receiving
                    # the packet again.
                    if (hasattr(error, 'errno') and error.errno ==
                       errno.ECONNRESET) or self.recv_socket is None:
                        self._retry_connection()

                    warnings.warn("UDPSocket Error @ t = " + str(t) +
                                  "s: " + str(error))

        # Return retrieved value
        return self.value

    def close(self):
        self._terminate_alive_check_thread()
        self._close_send_socket()
        self._close_recv_socket()
