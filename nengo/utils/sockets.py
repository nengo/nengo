from __future__ import absolute_import

import struct
import socket
import time
import threading
import errno
import warnings

from nengo.utils.compat import queue


class SocketCheckAliveThread(threading.Thread):
    """Check UDPSocket class for inactivity, and close if a message has
    not been sent or received in the specified time.

    Parameters
    ----------
    socket_class : UDPSocket
        Monitor the activity of this UDPSocket class.
    """
    def __init__(self, socket_class):
        threading.Thread.__init__(self)
        self.socket_class = socket_class

    def run(self):
        # Keep checking if the socket class is still being used.
        while (time.time() - self.socket_class.last_active <
               self.socket_class.idle_time_limit):
            time.sleep(self.socket_class.alive_thread_sleep_time)
        # If the socket class is idle, terminate the sockets
        self.socket_class._close_recv_socket()
        self.socket_class._close_send_socket()


class UDPSocket(object):
    """ A class for UDP communication to/from a Nengo model.

    A UDPSocket can be send only, receive only, or both send and receive.
    For each of these cases, different parameter sets must be specified.

    If the ``local_addr`` or ``dest_addr`` are not specified, then a local
    connection is assumed.

    For a send only socket, the user must specify:
        (send_dim, dest_port)
    and may optionally specify:
        (dest_addr)

    For a receive only socket, the user must specify:
        (recv_dim, local_port)
    and may optionally specify:
        (local_addr, socket_timeout, thread_timeout)

    For a send and receive socket, the user must specify:
        (send_dim, recv_dim, local_port, dest_port)
    and may optionally specify:
        (local_addr, dest_addr, dt_remote, socket_timeout, thread_timeout)

    For examples of the UDPSocket communicating between models all running
    on a local machine, please see the tests/test_socket.py file.

    To communicate between two models in send and receive mode over a network,
    one running on machine A with IP address 10.10.21.1 and one running on
    machine B, with IP address 10.10.21.25, we add the following socket to the
    model on machine A::

        socket_send_recv_A = UDPSocket(
            send_dim=A_output_dims, recv_dim=B_output_dims,
            local_addr='10.10.21.1', local_port=5001,
            dest_addr='10.10.21.25', dest_port=5002)
        node_send_recv_A = nengo.Node(
            socket_send_recv_A.run,
            size_in=A_output_dims,  # input to this node is data to send
            size_out=B_output_dims)  # output from this node is data received

    and the following socket on machine B::

        socket_send_recv_B = UDPSocket(
            send_dim=B_output_dims, recv_dim=A_output_dims,
            local_addr='10.10.21.25', local_port=5002,
            dest_addr='10.10.21.1', dest_port=5001)
        node_send_recv_B = nengo.Node(
            socket_send_recv_B.run,
            size_in=B_output_dims,  # input to this node is data to send
            size_out=A_output_dims)  # output from this node is data received

    and then connect the ``UDPSocket.input`` and ``UDPSocket.output`` nodes to
    the communicating Nengo model elements.

    Parameters
    ----------
    send_dim : int, optional (Default: 1)
        Number of dimensions of the vector data being sent.
    recv_dim : int, optional (Default: 1)
        Number of dimensions of the vector data being received.
    dt_remote : float, optional (Default: 0)
        The time step of the remote simulation, only relevant for send and
        receive nodes. Used to regulate how often data is sent to the remote
        machine, handling cases where simulation time steps are not the same.
    local_addr : str, optional (Default: '127.0.0.1')
        The local IP address data is received over.
    local_port : int
        The local port data is receive over.
    dest_addr : str, optional (Default: '127.0.0.1')
        The local or remote IP address data is sent to.
    dest_port: int
        The local or remote port data is sent to.
    socket_timeout : float, optional (Default: 30)
        The time a socket waits before throwing an inactivity exception.
    thread_timeout : float, optional (Default: 1)
        The amount of inactive time allowed before closing a thread running
        a socket.
    byte_order : str, optional (Default: '!')
        Specify 'big' or 'little' endian data format.
        '!' uses the system default.
    """
    def __init__(self, send_dim=1, recv_dim=1, dt_remote=0,
                 local_addr='127.0.0.1', local_port=-1,
                 dest_addr='127.0.0.1', dest_port=-1,
                 socket_timeout=30, thread_timeout=1,
                 byte_order='!'):
        self.local_addr = local_addr
        self.local_port = local_port
        self.dest_addr = (dest_addr if isinstance(dest_addr, list)
                          else [dest_addr])
        self.dest_port = (dest_port if isinstance(dest_port, list)
                          else [dest_port])
        self.socket_timeout = socket_timeout

        if byte_order.lower() == "little":
            self.byte_order = '<'
        elif byte_order.lower() == "big":
            self.byte_order = '>'
        else:
            self.byte_order = byte_order

        self.last_t = 0.0  # local sim time last time run was called
        self.last_packet_t = 0.0  # remote sim time from last packet received
        self.dt = 0.0   # local simulation dt
        self.dt_remote = max(dt_remote, self.dt)  # dt between each packet sent

        self.last_active = time.time()
        self.idle_time_limit_min = thread_timeout
        self.idle_time_limit_max = max(thread_timeout, socket_timeout + 1)
        # threshold time for closing inactive socket thread
        self.idle_time_limit = self.idle_time_limit_max
        self.alive_check_thread = None
        # how often to check the socket threads for inactivity
        self.alive_thread_sleep_time = thread_timeout / 2.0
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
        """Reset state of socket, empty queue of messages."""
        self.value = [0.0] * self.recv_dim
        self.last_t = 0.0
        self.last_packet_t = 0.0

        # Empty the buffer
        while not self.buffer.empty():
            self.buffer.get()

    def _open_socket(self):
        """Startup socket and thread for communication inactivity."""
        # Close socket, terminate alive check thread
        self.close()

        if self.is_sender:
            try:
                self.send_socket = socket.socket(socket.AF_INET,
                                                 socket.SOCK_DGRAM)
                if self.dest_addr == self.local_addr:
                    self.send_socket.bind((self.local_addr, 0))
            except socket.error as error:
                raise RuntimeError("UDPSocket: Error str: " + str(error))

        if self.is_receiver:
            self._open_recv_socket()

        self.last_active = time.time()
        self.alive_check_thread = SocketCheckAliveThread(self)
        self.alive_check_thread.start()

    def _open_recv_socket(self):
        """Create a socket for receiving data."""
        try:
            self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.recv_socket.bind((self.local_addr, self.local_port))
            self.recv_socket.settimeout(self.socket_timeout)
        except socket.error:
            raise RuntimeError("UDPSocket: Could not bind to socket. "
                               "Address: %s, Port: %s, is in use. If "
                               "simulation has been run before, wait for "
                               "previous UDPSocket to release the port. "
                               "(See 'max_idle_time' argument in class "
                               "constructor, currently set to %f seconds)" %
                               (self.local_addr, self.local_port,
                                self.idle_time_limit))

    def _retry_connection(self):
        """Close any open receive sockets, try to create a new receive
        socket with increasing delays between attempts."""
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
        """Set up situation for the alive_check_thread to shut down
        this class due to communication inactivity and exit."""
        self.idle_time_limit = self.idle_time_limit_max
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

    def pack_packet(self, t, x):
        """Takes a time stamp and data (x) and makes a socket packet

        Default packet data type: float
        Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]"""

        send_data = [float(t + self.dt_remote / 2.0)] + \
                    [x[i] for i in range(self.send_dim)]
        packet = struct.pack(self.byte_order + 'f' * (self.send_dim + 1),
                             *send_data)
        return packet

    def unpack_packet(self, packet):
        """Takes a packet and extracts a time stamp and data (x)

        Default packet data type: float
        Default packet structure: [t, x[0], x[1], x[2], ... , x[d]]"""

        data_len = len(packet) / 4
        data = list(struct.unpack(self.byte_order + 'f' * data_len, packet))
        t_data = data[0]
        value = data[1:]
        return t_data, value

    def __call__(self, t, x=None):
        return self.run(t, x)

    def _t_check(self, t_lim, t):
        """Check to see if current or next time step is closer to t_lim."""
        return (t_lim >= t and t_lim < t + self.dt) or self.ignore_timestamp

    def run(self, t, x=None):  # noqa: C901
        """Function to pass into Nengo node. In case of both sending and
        receiving the sending frequency is regulated by comparing the local
        and remote time steps. Information is sent when the current local
        time step is closer to the remote time step than the next local time
        step.
        """
        # If t == 0, return array of zeros and terminate
        # any open sockets to reset system
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
        # most often that an update can be sent is every self.dt,
        # so if remote dt is smaller just use self.dt for check
        self.dt_remote = max(self.dt_remote, self.dt)
        self.last_t = t * 1.0
        self.last_active = time.time()

        if self.is_sender:
            # Calculate if it is time to send the next packet.
            # Ideal time to send is last_packet_t + dt_remote, and we
            # want to find out if current or next local time step is closest.
            if (t + self.dt / 2.0) >= (self.last_packet_t + self.dt_remote):
                for addr in self.dest_addr:
                    for port in self.dest_port:
                        self.send_socket.sendto(self.pack_packet(t * 1.0, x),
                                                (addr, port))
                self.last_packet_t = t * 1.0  # Copy t (which is a scalar)

        if self.is_receiver:
            found_item = False
            if not self.buffer.empty():
                # There are items (packets with future timestamps) in the
                # buffer. Check the buffer for appropriate information
                t_peek = self.buffer.queue[0][0]
                if self._t_check(t_peek, t):
                    # Time stamp of first item in buffer is >= t and < t+dt,
                    # meaning that this is the information for the current
                    # time step, so it should be used.
                    data = self.buffer.get()
                    self.value = data[1]
                    found_item = True
                elif (t_peek >= t + self.dt):
                    # Time stamp of first item in buffer is > t+dt (i.e. all
                    # items in the buffer are future packets). Assume packet
                    # for current time step has been lost, don't read the info,
                    # and wait for local sim time to catch up.
                    found_item = True

            while not found_item:
                try:
                    packet, addr = self.recv_socket.recvfrom(self.max_recv_len)
                    t_data, value = self.unpack_packet(packet)
                    if self._t_check(t_data, t):
                        self.value = value
                        found_item = True
                    elif (t_data >= t + self.dt):
                        self.buffer.put((t_data, value))
                        found_item = True

                    # Packet recv success! Decay idle_time_limit time
                    # to the user specified max_idle_time (which can be
                    # smaller than the socket timeout time)
                    self.idle_time_limit = \
                        max(self.idle_time_limit_min,
                            self.idle_time_limit * 0.9)
                    self.retry_backoff_time = \
                        max(1, self.retry_backoff_time / 2)

                except (socket.error, AttributeError) as error:
                    found_item = False

                    # Socket error has occurred. Probably a timeout.
                    # Assume worst case, set idle_time_limit to
                    # idle_time_limit_max to wait for more timeouts to
                    # occur (this is so that the socket isn't constantly
                    # closed by the check_alive thread)
                    self.idle_time_limit = self.idle_time_limit_max

                    # Timeout occurred, assume packet lost.
                    if isinstance(error, socket.timeout):
                        found_item = True

                    # If connection was reset (somehow?), or closed by the
                    # idle timer (prematurely), retry the connection, and
                    # retry receiving the packet again.
                    if (hasattr(error, 'errno') and error.errno ==
                       errno.ECONNRESET) or self.recv_socket is None:
                        self._retry_connection()

                    warnings.warn("UDPSocket Error @ t = " + str(t) +
                                  "s: " + str(error))

        # Return retrieved value
        return self.value

    def close(self):
        """Close all threads and sockets."""
        self._terminate_alive_check_thread()
        # Double make sure all sockets are closed
        self._close_send_socket()
        self._close_recv_socket()
