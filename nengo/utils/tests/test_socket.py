import threading
import numpy as np

import nengo
from nengo.utils import sockets


class SimThread(threading.Thread):
    def __init__(self, simulator, sim_time, lock=None, wait=False):
        threading.Thread.__init__(self)
        self.sim = simulator
        self.sim_time = sim_time
        self.lock = lock
        self.wait = wait

    def run(self):
        if self.lock is not None and self.wait:
            self.lock.wait()
        if self.lock is not None and not self.wait:
            self.lock.set()
        self.sim.run(self.sim_time)


def test_send_recv_chain(Simulator, plt, seed, rng):
    # Model that sends data
    udp_send = sockets.UDPSocket(dest_port=54321, send_dim=1)
    m_send = nengo.Network(label='Send', seed=seed)
    with m_send:
        input = nengo.Node(output=lambda t: np.sin(10 * t))
        socket_node = nengo.Node(size_in=1, output=udp_send.run)

        nengo.Connection(input, socket_node, synapse=None)
        p_s = nengo.Probe(input, synapse=None)
    sim_send = Simulator(m_send)

    # Model that receives data from previous model, feeds data back to itself,
    # and sends that data out again
    udp_both = sockets.UDPSocket(dest_port=54322, local_port=54321,
                                 send_dim=1, recv_dim=1, timeout=1)
    m_both = nengo.Network(label='Both', seed=seed)
    with m_both:
        socket_node = nengo.Node(size_in=1, output=udp_both.run)

        nengo.Connection(socket_node, socket_node, synapse=0)
        p_b = nengo.Probe(socket_node, synapse=None)
    sim_both = Simulator(m_both)

    # Model that receives data from previous model
    udp_recv = sockets.UDPSocket(local_port=54322, recv_dim=1, timeout=1)
    m_recv = nengo.Network(label='Recv', seed=seed)
    with m_recv:
        socket_node = nengo.Node(output=udp_recv.run)
        p_r = nengo.Probe(socket_node, synapse=None)
    sim_recv = Simulator(m_recv)

    # Create thread lock to lock sending thread until receiving thread is ready
    thread_lock = threading.Event()

    # Create threads to run simulations
    sim_thread_send = SimThread(sim_send, 0.5, thread_lock, True)
    sim_thread_both = SimThread(sim_both, 0.5, thread_lock, True)
    sim_thread_recv = SimThread(sim_recv, 0.5, thread_lock, False)
    sim_thread_recv.start()
    sim_thread_both.start()
    sim_thread_send.start()
    sim_thread_send.join()
    sim_thread_both.join()
    sim_thread_recv.join()

    # Cleanup sockets
    udp_send.close()
    udp_both.close()
    udp_recv.close()

    # Do plots
    plt.subplot(3, 1, 1)
    plt.plot(sim_send.trange(), sim_send.data[p_s])
    plt.title('1: Send node. Sends to 2.')
    plt.subplot(3, 1, 2)
    plt.plot(sim_both.trange(), sim_both.data[p_b])
    plt.title('2: Send and recv node. Recvs from 1, sends to 3.')
    plt.subplot(3, 1, 3)
    plt.plot(sim_recv.trange(), sim_recv.data[p_r])
    plt.title('3: Recv node. Recvs from 2.')

    # Note: The socket communication delays information by 1 timestep in m_both
    assert np.allclose(sim_send.data[p_s], sim_both.data[p_b],
                       atol=0.0001, rtol=0.0001)
    assert np.allclose(sim_both.data[p_b][:-1], sim_recv.data[p_r][1:],
                       atol=0.0001, rtol=0.0001)


def test_time_sync(Simulator, plt, seed, rng):
    udp1 = sockets.UDPSocket(dest_port=54322, local_port=54321,
                             send_dim=1, recv_dim=2, timeout=1)
    m1 = nengo.Network(label='One', seed=seed)
    with m1:
        input = nengo.Node(output=lambda t: np.sin(10 * t))
        socket_node = nengo.Node(size_in=1, output=udp1.run)

        nengo.Connection(input, socket_node, synapse=None)
        p_i1 = nengo.Probe(input, synapse=None)
        p_s1 = nengo.Probe(socket_node, synapse=None)
    sim1 = Simulator(m1)

    # Model that receives data from previous model
    udp2 = sockets.UDPSocket(dest_port=54321, local_port=54322,
                             send_dim=2, recv_dim=1, timeout=1)
    m2 = nengo.Network(label='Two', seed=seed)
    with m2:
        input = nengo.Node(output=lambda t: [math.cos(10 * t), t])
        socket_node = nengo.Node(size_in=2, output=udp2.run)

        nengo.Connection(input, socket_node, synapse=None)
        p_i2 = nengo.Probe(input, synapse=None)
        p_s2 = nengo.Probe(socket_node, synapse=None)
    sim2 = Simulator(m2)

    # Create threads to run simulations
    sim_thread1 = SimThread(sim1, 0.5)
    sim_thread2 = SimThread(sim2, 0.5)
    sim_thread1.start()
    sim_thread2.start()
    sim_thread1.join()
    sim_thread2.join()

    # Cleanup sockets
    udp1.close()
    udp2.close()

    # Do plots
    plt.subplot(4, 1, 1)
    plt.plot(sim1.trange(), sim1.data[p_i1])
    plt.title('Input to Node 1. Sent to Node 2.')
    plt.subplot(4, 1, 2)
    plt.plot(sim2.trange(), sim2.data[p_s2])
    plt.title('Output from Node 2. Recvs from Node 1.')

    plt.subplot(4, 1, 3)
    plt.title('Input to Node 2. Sent to Node 1.')
    plt.plot(sim2.trange(), sim2.data[p_i2])
    plt.subplot(4, 1, 4)
    plt.title('Output from Node 1. Recvs from Node 2.')
    plt.plot(sim1.trange(), sim1.data[p_s1])

    # Test results
    assert np.allclose(sim1.data[p_i1], sim2.data[p_s2],
                       atol=0.0001, rtol=0.0001)
    assert np.allclose(sim2.data[p_i2], sim1.data[p_s1],
                       atol=0.0001, rtol=0.0001)
