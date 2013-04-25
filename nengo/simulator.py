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
        self.old_state = {}
        self.new_state = {}

    def _build(self, dt):
        for node in self.nengo_objects:
            node._build(self.old_state, dt)

    def reset(self):
        for cc in self.nengo_objects:
            cc._reset(self.old_state)

    def run(self, simtime, dt, stop_when=None, dump_probes_fn=None):      
        n_steps = int(simtime / dt)

        for ii in xrange(n_steps):
            for cc in self.nengo_objects:
                cc._step(self.old_state, self.new_state, dt)
            
            self.old_state = self.new_state
            self.new_state = {}
            
            if stop_when and stop_when():
                break
                
        if dump_probes_fn:
            return dump_probes_fn(self.network.probes)

