"""
object_api_python.py

Pure-python implementation of models described by the classes in object_api.py.

"""

import copy
from functools import partial
import sys
import math
import random

import networkx as nx

import object_api as API


class SimulatorException(Exception):
    pass

class ResetIncomplete(SimulatorException):
    pass

class StepIncomplete(SimulatorException):
    pass

class SizeError(SimulatorException):
    """Implementation produced output of incorrect size"""



def scheduling_graph(network):
    all_members = network.all_members
    DG = nx.DiGraph()
    producer = {}
    for member in all_members:
        for key, val in member.inputs.items():
            if val is None:
                pass
            elif isinstance(val, API.DelayedVar):
                pass
            else:
                DG.add_edge(val, member)
        for key, val in member.outputs.items():
            if val is None:
                raise ValueError('Output cannot be None', key)
            elif isinstance(val, API.DelayedVar):
                raise ValueError('Output cannot be delayed', key)
            else:
                if producer.setdefault(val, member) is not member:
                    raise API.MultipleSourceError(val)
                DG.add_edge(member, val)
    return DG


class ImplBase(object):
    @staticmethod
    def build(obj, state, dt):
        pass
    @staticmethod
    def reset(obj, state):
        pass
    @staticmethod
    def step(obj, old_state, new_state):
        pass
    @staticmethod
    def probe(obj, state):
        pass
    @staticmethod
    def sample(obj, N):
        pass


class SimulatorState(object):
    def __init__(self):
        self.new_state = {}
        self.old_state = {}
        self.probes = {}

    def __contains__(self, item):
        return item in self.new_state

    def __getitem__(self, item):
        try:
            return self.old_state[item.var]
        except AttributeError:
            return self.new_state[item]

    def __setitem__(self, item, val):
        self.new_state[item] = val

    def step(self):
        self.old_state = self.new_state
        self.new_state = {}



class Simulator(API.SimulatorBase):

    impl_registry = {}
    build_registry = {}
    reset_registry = {}
    step_registry = {}
    sample_registry = {}

    @classmethod
    def register_impl(mycls, cls):
        api_cls = getattr(API, cls.__name__)
        mycls.build_registry[api_cls] = cls.build
        mycls.reset_registry[api_cls] = cls.reset
        mycls.step_registry[api_cls] = cls.step
        mycls.sample_registry[api_cls] = cls.sample
        return cls

    def __init__(self, network, dt, verbosity=0):
        API.SimulatorBase.__init__(self, network)
        self.state = SimulatorState()
        self.dt = dt
        self.verbosity = verbosity
        self.state[API.simulation_time] = self.simulation_time
        DG = scheduling_graph(network)
        #import matplotlib.pyplot as plt
        #nx.draw(DG)
        #plt.show()
        try:
            self.full_ordering = nx.topological_sort(DG)
        except nx.NetworkXUnfeasible:
            #nx.write_multiline_adjlist(DG, sys.stdout)
            raise API.SelfDependencyError()
        self.member_ordering = [n for n in self.full_ordering
                              if not isinstance(n, API.Var)]

        for member in self.member_ordering:
            build_fn = self.build_registry.get(type(member), None)
            if build_fn:
                if verbosity:
                    print 'Build:', member, build_fn, verbosity
                build_fn(member, self.state, self.dt)
            elif verbosity:
                print 'No build:', member
        self.reset()

    def reset(self):
        API.SimulatorBase.reset(self)
        self.state[API.simulation_time] = self.simulation_time
        for member in self.member_ordering:
            reset_fn = self.reset_registry.get(type(member), None)
            if reset_fn:
                if self.verbosity:
                    print 'Reset:', member, reset_fn
                reset_fn(member, self.state)
            elif self.verbosity:
                print 'No reset:', member
            for key, val in member.outputs.items():
                if val not in self.state:
                    raise ResetIncomplete("Reset %s did not produce outputs[%s]" % (
                        member, key))

    def run_steps(self, steps):
        step_fns = []
        for member in self.member_ordering:
            step_fn = self.step_registry.get(type(member), None)
            if step_fn:
                if self.verbosity:
                    print 'Step:', member, step_fn
                step_fns.append((step_fn, member))
            elif self.verbosity:
                print 'No step:', member

        for tt in xrange(steps):
            self.simulation_time += self.dt
            self.state.step()
            self.state[API.simulation_time] = self.simulation_time
            for step_fn, member in step_fns:
                try:
                    step_fn(member, self.state)
                except TypeError, e:
                    if "takes exactly" in str(e):
                        e.args = e.args + (step_fn, 'of implementation for',  member)
                    raise
                for key, val in member.outputs.items():
                    if val not in self.state:
                        raise StepIncomplete("Step %s did not produce outputs[%s]" % (
                            member, key))
                    if self.state[val].size != val.size:
                        raise SizeError(
                            "Step %s produced outputs[%s] of wrong size %i, which"
                            " should have been %i" % (
                            member, key, self.state[val].size, val.size))

        rval = {}
        for probe in self.network.all_probes:
            # -- probe_fn is mandatory
            rval[probe.target] = self.state.probes[probe]
            self.state.probes[probe] = []

        return rval

    def run(self, sim_time):
        steps = int(sim_time / self.dt)
        return self.run_steps(steps)

API.SimulatorBase._backends['reference'] = Simulator
register_impl = Simulator.register_impl


#
# Pure Python Object API implementation
#

@register_impl
class TimeNode(ImplBase):
    @staticmethod
    def build(node, state, dt):
        state[node.output] = node.func(state[API.simulation_time])

    @staticmethod
    def reset(node, state):
        state[node.output] = node.func(state[API.simulation_time])

    @staticmethod
    def step(node, old_state, new_state):
        t = new_state[API.simulation_time]
        new_state[node.output] = node.func(t)


@register_impl
class Probe(ImplBase):
    @staticmethod
    def reset(probe, state):
        state.probes[probe] = []

    @staticmethod
    def step(probe, state):
        obj = copy.deepcopy(state[probe.target])
        state.probes[probe].append(obj)


@register_impl
class Uniform(ImplBase):
    @staticmethod
    def build(dist, state, dt):
        rng = random.Random(dist.seed)
        def draw_n(N):
            return[rng.uniform(dist.low, dist.high) for ii in xrange(N)]
        state[dist.rng] = draw_n

    @staticmethod
    def step(obj, old_state, new_state):
        new_state[obj.rng] = old_state[obj.rng]


@register_impl
class Gaussian(ImplBase):
    @staticmethod
    def build(dist, state, dt):
        rng = random.Random(dist.seed)
        def draw_n(N):
            return[rng.gauss(dist.mean, dist.std) for ii in xrange(N)]
        state[dist.rng] = draw_n

    @staticmethod
    def step(obj, old_state, new_state):
        new_state[obj.rng] = old_state[obj.rng]


@register_impl
class LIFNeurons(ImplBase):
    @staticmethod
    def build(neurons, state, dt):
        build(neurons.max_rate, state, dt)
        build(neurons.intercept, state, dt)

        max_rates = state[neurons.max_rate.rng](neurons.size)
        threshold = state[neurons.intercept.rng](neurons.size)

        def x_fn(max_rate):
            u = neurons.tau_ref - (1.0 / max_rate)
            return 1.0 / (1 - math.exp(u / neurons.tau_rc))
        xlist = map(x_fn, max_rates)
        alpha = [(1 - x) / intercept for x, intercept in zip(xlist, threshold)]
        j_bias = [1 - aa * intercept for aa, intercept in zip(alpha, threshold)]

        state[neurons.alpha] = alpha
        state[neurons.j_bias] = j_bias
        state[neurons.voltage] = [0] * neurons.size
        state[neurons.refractory_time] = [0] * neurons.size
        state[neurons.output] = [0] * neurons.size

    @staticmethod
    def reset(neurons, state):
        state[neurons.voltage] = [0] * neurons.size
        state[neurons.refractory_time] = [0] * neurons.size
        state[neurons.output] = [0] * neurons.size

    @staticmethod
    def step(neurons, old_state, new_state):
        alpha = old_state[neurons.alpha]
        j_bias = old_state[neurons.j_bias]
        voltage = old_state[neurons.voltage]
        refractory_time = old_state[neurons.refractory_time]
        tau_rc = neurons.tau_rc
        tau_ref = neurons.tau_ref
        dt = new_state[API.simulation_time] - old_state[API.simulation_time]
        J  = j_bias # XXX WRONG MATH
        new_voltage = [0] * neurons.size
        new_refractory_time = [0] * neurons.size
        new_output = [0] * neurons.size

        def clip(a, low, high):
            if a < low:
                return low
            if a > high:
                return high
            return a

        for ii in xrange(neurons.size):

            # Euler's method
            dV = dt / tau_rc * (J[ii] - voltage[ii])

            # increase the voltage, ignore values below 0
            v = max(voltage[ii] + dV, 0)

            # handle refractory period
            post_ref = 1.0 - (refractory_time[ii] - dt) / dt

            # set any post_ref elements < 0 = 0, and > 1 = 1
            if post_ref < 0:
                v = 0
            elif post_ref < 1:
                v *= post_ref

            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked = 1 if v > 1 else 0

            # adjust refractory time (neurons that spike get
            # a new refractory time set, all others get it reduced by dt)

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            if spiked:
                new_refractory_time[ii] = spiketime + tau_ref
            else:
                new_refractory_time[ii] = refractory_time[ii] - dt

            new_voltage[ii] = v * (1 - spiked)
            new_output[ii] = spiked


        new_state[neurons.alpha] = alpha
        new_state[neurons.j_bias] = j_bias
        new_state[neurons.voltage] = new_voltage
        new_state[neurons.refractory_time] = new_refractory_time
        new_state[neurons.output] = new_output


