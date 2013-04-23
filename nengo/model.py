import random
import collections
#import quantities
import numpy as np
from simulator import Simulator
from network import Network

class Model(object):
    
    def __init__(self, name, seed=None, fixed_seed=None):
        """Container class that binds a Network model to a simulator for execution.

        :param string name:
            create and wrap a new Network with the given name.
        :param int seed:
            random number seed to use for creating ensembles.
            This one seed is used only to start the
            random generation process, so each neural group
            created will be different.

        :param int fixed_seed:
            random number seed for creating ensembles
            this one seed is used for all ensembles to create thier neurons
            this means that the neurons will have the same xintercepts and firing rates
            for all ensembles (different from seed above)
            
        """
        self.network = Network(name)
        self.name = name

        self.time = 0

        self.backend_type = ''

    #
    # Execution methods
    #

    def build(self, dt=0.001):
        self.simulator = Simulator(self.network)
        self.simulator.build(dt)

    
    def reset(self):
        """ Reset the state of the simulation

            Runs through all nodes, then ensembles, then connections and then
            probes in the network and calls thier reset functions
            
        """
        self.simulator.reset()

    def run(self, time, dt=0.001, output=None, stop_when=None):
        """Run the simulation.

        If called twice, the simulation will continue for *time*
        more seconds. Note that the ensembles are simulated at the
        dt timestep specified when they are created.
        
        :param float time: the amount of time (in seconds) to run
        :param float dt: the timestep of the update
        
        """
        self.simulator.step_time(time, dt, stop_when=stop_when,
                                dump_probes_fn=output)

                    
    # Wrappers for Network methods
    def add(self, node):
        return self.network.add(node)

    def compute_transform(self, dim_pre, dim_post, array_size, weight=1,
                          index_pre=None, index_post=None, transform=None):
        return self.network.compute_transform(
            dim_pre, dim_post, array_size, weight,
            index_pre, index_post, transform )
                
    def connect(self, pre, post, transform=None, weight=1,
                index_pre=None, index_post=None, pstc=0.01, 
                func=None):
        return self.network.connect(
            pre, post, transform, weight, index_pre, index_post, pstc, func)
    
    def get_object(self, name):
            return self.network.get_object(name)
       
    def get_origin(self, name, func=None):
        return self.network.get_origin(name, func)

    def learn(self, pre, post, error, pstc=0.01, **kwargs):
        return self.network.learn(pre, post, error, pstc, **kwargs);
                                  
    def make_ensemble(self, name, *args, **kwargs): 
        return self.network.make_ensemble(name, *args, **kwargs)

    def make_array(self, name, neurons, array_size, dimensions=1, **kwargs):
        return self.network.make_array(name, neurons, array_size, dimensions, **kwargs)
    
    def make_node(self, *args, **kwargs): 
        """
        XXX
        Don't you wish the docs were here?
        """
        return self.network.make_node(*args, **kwargs)
        
    def make_subnetwork(self, name):
        return self.network.make_subnetwork(name)
            
    def probe(self, *args, **kwargs):
        return self.network.probe(*args, **kwargs)

                                
