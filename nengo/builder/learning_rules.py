import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import DotInc, ElementwiseInc, Operator, Reset
from nengo.builder.signal import Signal
from nengo.builder.synapses import filtered_signal
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.learning_rules import BCM, Oja, PES


class SimBCM(Operator):
    """Change the transform according to the BCM rule."""
    def __init__(self, pre_filtered, post_filtered, theta, delta,
                 learning_rate):
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.theta = theta
        self.delta = delta
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, theta]
        self.updates = [delta]

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - theta), pre_filtered)
        return step


class SimOja(Operator):
    """
    Change the transform according to the OJA rule
    """
    def __init__(self, pre_filtered, post_filtered, transform, delta,
                 learning_rate, beta):
        self.post_filtered = post_filtered
        self.pre_filtered = pre_filtered
        self.transform = transform
        self.delta = delta
        self.learning_rate = learning_rate
        self.beta = beta

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, transform]
        self.updates = [delta]

    def make_step(self, signals, dt, rng):
        transform = signals[self.transform]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt
        beta = self.beta

        def step():
            # perform forgetting
            post_squared = alpha * post_filtered * post_filtered
            delta[...] = -beta * transform * post_squared[:, None]

            # perform update
            delta[...] += np.outer(alpha * post_filtered, pre_filtered)

        return step


def build_learning_rule(rule, model, config):
    rule_type = rule.learning_rule_type
    Builder.build(rule_type, rule, model=model, config=config)

Builder.register_builder(build_learning_rule, LearningRule)


def build_bcm(bcm, rule, model, config):
    conn = rule.connection
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    transform = model.sig[conn]['transform']
    pre_activities = model.sig[pre.neurons]['out']
    post_activities = model.sig[post.neurons]['out']
    pre_filtered = filtered_signal(
        bcm, pre_activities, bcm.pre_tau, model, config)
    post_filtered = filtered_signal(
        bcm, post_activities, bcm.post_tau, model, config)
    theta = filtered_signal(
        bcm, post_filtered, bcm.theta_tau, model, config)
    delta = Signal(np.zeros((post.n_neurons, pre.n_neurons)),
                   name='BCM: Delta')

    model.add_op(SimBCM(pre_filtered, post_filtered, theta, delta,
                        learning_rate=bcm.learning_rate))
    model.add_op(ElementwiseInc(
        model.sig['common'][1], delta, transform, tag="BCM: Inc Transform"))

    model.params[rule] = None

Builder.register_builder(build_bcm, BCM)


def build_oja(oja, rule, model, config):
    conn = rule.connection
    pre = (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
           else conn.pre_obj.ensemble)
    post = (conn.post_obj if isinstance(conn.post_obj, Ensemble)
            else conn.post_obj.ensemble)
    transform = model.sig[conn]['transform']
    pre_activities = model.sig[pre.neurons]['out']
    post_activities = model.sig[post.neurons]['out']
    pre_filtered = filtered_signal(
        oja, pre_activities, oja.pre_tau, model, config)
    post_filtered = filtered_signal(
        oja, post_activities, oja.post_tau, model, config)
    delta = Signal(np.zeros((post.n_neurons, pre.n_neurons)),
                   name='Oja: Delta')

    model.add_op(SimOja(pre_filtered, post_filtered, transform, delta,
                        learning_rate=oja.learning_rate, beta=oja.beta))
    model.add_op(ElementwiseInc(
        model.sig['common'][1], delta, transform, tag="Oja: Inc Transform"))

    model.params[rule] = None

Builder.register_builder(build_oja, Oja)


def build_pes(pes, rule, model, config):
    conn = rule.connection
    activities = model.sig[conn.pre_obj]['out']
    error = model.sig[pes.error_connection]['out']

    scaled_error = Signal(np.zeros(error.shape),
                          name="PES:error * learning_rate")
    scaled_error_view = scaled_error.reshape((error.size, 1))
    activities_view = activities.reshape((1, activities.size))

    model.sig[rule]['scaled_error'] = scaled_error
    model.sig[rule]['activities'] = activities

    lr_sig = Signal(pes.learning_rate * model.dt, name="PES:learning_rate")

    model.add_op(Reset(scaled_error))
    model.add_op(DotInc(lr_sig, error, scaled_error, tag="PES:scale error"))

    if conn.solver.weights or (
            isinstance(conn.pre_obj, Neurons) and
            isinstance(conn.post_obj, Neurons)):
        post = (conn.post_obj.ensemble if isinstance(conn.post_obj, Neurons)
                else conn.post_obj)
        transform = model.sig[conn]['transform']
        encoders = model.sig[post]['encoders']
        encoded_error = Signal(np.zeros(transform.shape[0]),
                               name="PES: encoded error")

        model.add_op(Reset(encoded_error))
        model.add_op(DotInc(
            encoders, scaled_error, encoded_error, tag="PES:Encode error"))

        encoded_error_view = encoded_error.reshape((encoded_error.size, 1))
        model.add_op(ElementwiseInc(
            encoded_error_view, activities_view, transform,
            tag="PES:Inc Transform"))
    elif isinstance(conn.pre_obj, Neurons):
        transform = model.sig[conn]['transform']
        model.add_op(ElementwiseInc(
            scaled_error_view, activities_view, transform,
            tag="PES:Inc Transform"))
    else:
        assert isinstance(conn.pre_obj, Ensemble)
        decoders = model.sig[conn]['decoders']
        model.add_op(ElementwiseInc(
            scaled_error_view, activities_view, decoders,
            tag="PES:Inc Decoder"))

    model.params[rule] = None

Builder.register_builder(build_pes, PES)
