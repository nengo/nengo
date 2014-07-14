import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import DotInc, Operator, Reset
from nengo.builder.signal import Signal, SignalView
from nengo.builder.synapses import filtered_signal
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
    """
    Change the transform according to the OJA rule
    """
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


def build_bcm(bcm, conn, model, config):
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    pre_activities = model.sig[pre]['neuron_out']
    post_activities = model.sig[post]['neuron_out']

    delta = Signal(np.zeros((post.n_neurons, pre.n_neurons)), name='delta')

    pre_filtered = filtered_signal(
        bcm, pre_activities, bcm.pre_tau, model, config)
    post_filtered = filtered_signal(
        bcm, post_activities, bcm.post_tau, model, config)
    theta = filtered_signal(
        bcm, post_filtered, bcm.theta_tau, model, config)

    transform = model.sig[conn]['transform']

    model.add_op(DotInc(
        model.sig['common'][1], delta, transform, tag="BCM: DotInc"))

    model.add_op(SimBCM(delta=delta,
                        pre_filtered=pre_filtered,
                        post_filtered=post_filtered,
                        theta=theta,
                        learning_rate=bcm.learning_rate))

Builder.register_builder(build_bcm, BCM)


def build_oja(oja, conn, model, config):
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    pre_activities = model.sig[pre]['neuron_out']
    post_activities = model.sig[post]['neuron_out']
    pre_filtered = filtered_signal(
        oja, pre_activities, oja.pre_tau, model, config)
    post_filtered = filtered_signal(
        oja, post_activities, oja.post_tau, model, config)
    omega_shape = (post.n_neurons, pre.n_neurons)

    transform = model.sig[conn]['transform']
    delta = Signal(np.zeros(omega_shape), name='Oja: Delta')
    forgetting = Signal(np.zeros(omega_shape), name='Oja: Forgetting')

    model.add_op(DotInc(
        model.sig['common'][1], delta, transform, tag="Oja: Delta DotInc"))

    model.add_op(DotInc(Signal(-oja.beta, "Oja: Negative oja scale"),
                        forgetting,
                        transform,
                        tag="Oja: Forgetting DotInc"))

    model.add_op(SimOja(transform=transform,
                        delta=delta,
                        pre_filtered=pre_filtered,
                        post_filtered=post_filtered,
                        forgetting=forgetting,
                        learning_rate=oja.learning_rate))

    model.params[oja] = None

Builder.register_builder(build_oja, Oja)


def build_pes(pes, conn, model, config):
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

    model.params[pes] = None

Builder.register_builder(build_pes, PES)
