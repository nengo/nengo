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

    def _reset(self):
        for obj in self.nengo_objects:
            obj._reset(self.old_state)

    def run(self, simtime, dt, stop_when=None):      
        n_steps = int(simtime / dt)

        for _ in range(n_steps):
            for obj in self.nengo_objects:
                obj._step(self.old_state, self.new_state, dt)
            
            self.old_state = self.new_state
            self.new_state = {}
            
            if stop_when and stop_when():
                break

