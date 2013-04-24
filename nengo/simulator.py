#try:
#    from collections import OrderedDict
#except:
#    from ordereddict import OrderedDict
#
#class SymbolicSignal(object):
#    def __init__(self,
#                 name=None, type=None, shape=None, units=None, dtype=None):
#        slef.name = name
#        self.type = type
#        self.shape = shape
#        self.units = units
#        self.dtype = dtype


def dump_probes_to_file(probes, filename):
    raise NotImplementedError()


def dump_probes_to_hdf5(probes, filename):
    raise NotImplementedError()


class Simulator(object):
    def __init__(self, network):
        self.network = network
        self.nengo_objects = (
            network.all_nodes + network.all_ensembles + network.all_probes)
        print self.nengo_objects
        self.state_t = {}
        self.state_tm1 = {}

    def _build(self, dt):
        for node in self.nengo_objects:
            node._build(self.state_t, dt)

    def reset(self):
        for cc in self.nengo_objects:
            cc._reset(self.state_t)

    def step_time(self, simtime, dt, stop_when=None, dump_probes_fn=None):
        if stop_when or dump_probes_fn:
            raise NotImplementedError()
        n_steps = int(simtime / dt)
        state_t = self.state_t
        state_tm1 = self.state_tm1
        for ii in xrange(n_steps):
            for cc in self.nengo_objects:
                cc._step(state_t, state_tm1, dt)
        if dump_probes_fn:
            return dump_probes_fn(self.network.probes)

