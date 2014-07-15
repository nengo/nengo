import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import DotInc, Operator, Reset
from nengo.builder.signal import Signal, SignalView
from nengo.connection import Connection
from nengo.ensemble import Ensemble, Neurons
from nengo.learning_rules import BCM, Oja, PES


class SimBCM(Operator):
    """Change the transform according to the BCM rule."""
    def __init__(self, delta,
                 pre_filtered, post_filtered, theta, learning_rate):
        self.delta = delta
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.theta = theta
        self.learning_rate = learning_rate

        self.reads = [theta, pre_filtered, post_filtered]
        self.updates = [delta]
        self.sets = []
        self.incs = []

    def make_step(self, signals, dt):
        delta = signals[self.delta]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        learning_rate = self.learning_rate

        def step():
            delta[...] = np.outer(post_filtered * (post_filtered - theta),
                                  pre_filtered) * learning_rate * dt
        return step


class SimOja(Operator):
    """Change the transform according to the Oja rule."""
    def __init__(self, transform, delta, pre_filtered, post_filtered,
                 forgetting, learning_rate):
        self.transform = transform
        self.delta = delta
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.forgetting = forgetting
        self.learning_rate = learning_rate

        self.reads = [transform, pre_filtered, post_filtered]
        self.updates = [delta, forgetting]
        self.sets = []
        self.incs = []

    def make_step(self, signals, dt):
        transform = signals[self.transform]
        delta = signals[self.delta]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        forgetting = signals[self.forgetting]
        learning_rate = self.learning_rate

        def step():
            post_squared = learning_rate * post_filtered * post_filtered
            for i in range(len(post_squared)):
                forgetting[i, :] = transform[i, :] * post_squared[i]

            delta[...] = np.outer(post_filtered, pre_filtered) * learning_rate
        return step


@Builder.register(Connection, BCM)
def build_bcm(model, conn, bcm):
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)

    delta = Signal(np.zeros((post.n_neurons, pre.n_neurons)), name='delta')
    transform = model.sig[conn]['transform']

    # Build the filters
    name = 'bcm%d' % id(bcm)  # Have to use the id here in case of multiples
    model.sig[conn]['%s_pre_in' % name] = model.sig[pre]['neuron_out']
    model.sig[conn]['%s_post_in' % name] = model.sig[post]['neuron_out']
    model.sig[conn]['%s_theta_in' % name] = model.sig[post]['neuron_out']
    model.build(conn, bcm.pre_synapse, '%s_pre' % name)
    model.build(conn, bcm.post_synapse, '%s_post' % name)
    model.build(conn, bcm.theta_synapse, '%s_theta' % name)

    model.add_op(DotInc(
        model.sig['common'][1], delta, transform, tag="BCM: DotInc"))
    model.add_op(SimBCM(delta=delta,
                        pre_filtered=model.sig[conn]['%s_pre_out' % name],
                        post_filtered=model.sig[conn]['%s_post_out' % name],
                        theta=model.sig[conn]['%s_theta_out' % name],
                        learning_rate=bcm.learning_rate))

    model.params[(conn, bcm)] = None


@Builder.register(Connection, Oja)
def build_oja(model, conn, oja):
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    transform = model.sig[conn]['transform']
    omega_shape = (post.n_neurons, pre.n_neurons)
    delta = Signal(np.zeros(omega_shape), name='Oja: Delta')
    forgetting = Signal(np.zeros(omega_shape), name='Oja: Forgetting')

    # Build the filters
    name = 'oja%d' % id(oja)  # Have to use the id here in case of multiples
    model.sig[conn]['%s_pre_in' % name] = model.sig[pre]['neuron_out']
    model.sig[conn]['%s_post_in' % name] = model.sig[post]['neuron_out']
    model.build(conn, oja.pre_synapse, '%s_pre' % name)
    model.build(conn, oja.post_synapse, '%s_post' % name)

    model.add_op(DotInc(
        model.sig['common'][1], delta, transform, tag="Oja: Delta DotInc"))
    model.add_op(DotInc(Signal(-oja.beta, "Oja: Negative oja scale"),
                        forgetting,
                        transform,
                        tag="Oja: Forgetting DotInc"))
    model.add_op(SimOja(transform=transform,
                        delta=delta,
                        pre_filtered=model.sig[conn]['%s_pre_out' % name],
                        post_filtered=model.sig[conn]['%s_post_out' % name],
                        forgetting=forgetting,
                        learning_rate=oja.learning_rate))

    model.params[(conn, oja)] = None


@Builder.register(Connection, PES)
def build_pes(model, conn, pes):
    # TODO: Filter activities

    if isinstance(conn.pre_obj, Neurons):
        activities = model.sig[conn.pre_obj.ensemble]['out']
    else:
        activities = model.sig[conn.pre_obj]['out']
    error = model.sig[pes.error_connection]['out']

    scaled_error = Signal(np.zeros(error.shape),
                          name="PES:error * learning rate")
    scaled_error_view = SignalView(scaled_error, (error.size, 1), (1, 1), 0,
                                   name="PES:scaled_error_view")
    activities_view = SignalView(activities, (1, activities.size), (1, 1), 0,
                                 name="PES:activities_view")

    lr_sig = Signal(pes.learning_rate * model.dt, name="PES:learning_rate")

    model.add_op(Reset(scaled_error))
    model.add_op(DotInc(lr_sig, error, scaled_error, tag="PES:scale error"))

    if conn.solver.weights or isinstance(conn.pre_obj, Neurons):
        outer_product = Signal(np.zeros((error.size, activities.size)),
                               name="PES: outer prod")
        transform = model.sig[conn]['transform']
        if isinstance(conn.post_obj, Neurons):
            encoders = model.sig[conn.post_obj.ensemble]['encoders']
        else:
            encoders = model.sig[conn.post_obj]['encoders']

        model.add_op(Reset(outer_product))
        model.add_op(DotInc(scaled_error_view, activities_view, outer_product,
                            tag="PES:Outer Prod"))
        model.add_op(DotInc(encoders, outer_product, transform,
                            tag="PES:Inc Decoder"))

    else:
        assert isinstance(conn.pre_obj, Ensemble)
        decoders = model.sig[conn]['decoders']

        model.add_op(DotInc(scaled_error_view, activities_view, decoders,
                            tag="PES:Inc Decoder"))

    model.params[(conn, pes)] = None
