"""
model_object.py

This file implements a Model class that conforms to the model API described
at API REF in terms of the objects described in object_api.py, which can, in turn
be implemented by multiple backends (so far: 'reference', 'numpy').

This file does not depend directly or indirectly on numpy, it should be safe to load it in
Jython.
"""
import collections
import quantities
import object_api as API

class Ensemble(object):
    def __init__(self, name, num_neurons, dimensions,
                 neuron_model, max_rate, intercept, seed,
                radius, encoders,):
        self.name = name
        self.dimensions = dimensions
        self.seed = seed
        self._adder = API.Adder(size=dimensions)
        if isinstance(neuron_model, LIFNeuron):
            rl, rh = max_rate
            il, ih = intercept
            self._neurons = API.LIFNeurons(num_neurons,
                max_rate=API.Uniform(rl, rh, seed=self.seed),
                intercept=API.Uniform(il, ih, seed=self.seed + 1)
                )
        else:
            raise NotImplementedError()

        if encoders is None:
            # XXX use radius!
            self._encoder = API.RandomConnection(self._adder.output, self._neurons.input_current,
                                                dist=API.Gaussian(0, 1, self.seed + 2))
        else:
            raise NotImplementedError()

    @property
    def num_neurons(self):
        return self._neurons.size


class LIFNeuron(object):
    def __init__(self, tau_ref=0.01, tau_rc=0.01):
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc


class Model(object):
    def __init__(self, name, seed=None, fixed_seed=None, backend_type='reference'):
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
        self.network = API.Network()
        self.byname = {}
        self.name = name
        self.backend_type = backend_type
        self.simulator = None
        if seed is None:
            self.seed = 123
        else:
            self.seed = seed
        if fixed_seed is not None:
            raise NotImplementedError()

    @property
    def time(self):
        return self.simulator.simulator_time

    def get_object(self, name):
        return self.get(name)
       
    def get_origin(self, name):
        return self.get(name)


    #
    # Execution methods
    #

    def build(self, dt=0.001):
        self.simulator = Simulator(self.network, dt=dt)
    
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
        if self.simulator is None:
            self.simulator = API.Simulator(self.network, dt=dt,
                                           backend=self.backend_type)
        assert dt == self.simulator.dt
        if stop_when is not None:
            raise NotImplementedError()
        if output is not None:
            raise NotImplementedError()
        steps = int(time / dt)
        self.simulator.run_steps(steps )

    def get(self, name):
        if name in self.byname:
            return self.byname[name]
        else:
            raise KeyError(name)

    def make_ensemble(self, name, num_neurons, dimensions,
                      max_rate=(50, 100),
                      intercept=(-1, 1),
                      radius=1,
                      encoders=None,
                      neuron_model=None,
                     ): 
        ens = self.byname[name] = Ensemble(name, num_neurons, dimensions,
                                           neuron_model=neuron_model,
                                           max_rate=max_rate,
                                           intercept=intercept,
                                           seed=self.seed,
                                           radius=radius,
                                           encoders=encoders,
                                          )
        self.network.add(ens._neurons)
        self.network.add(ens._encoder)
        self.network.add(ens._adder)
        self.seed += 101
        return ens

    def probe(self, target_name):
        """
        """
        obj = self.get_object(target_name)
        if isinstance(obj, Ensemble):
            return self.network.add(API.Probe(obj._adder.output))
        else:
            return self.network.add(API.Probe(obj))


    def _compute_transform(self, dim_pre, dim_post, array_size, weight=1,
                          index_pre=None, index_post=None, transform=None):
        """
        """
        return self.network.compute_transform(
            dim_pre, dim_post, array_size, weight,
            index_pre, index_post, transform )
                

    def _connect(self, pre, post, transform=None, weight=1,
                index_pre=None, index_post=None, pstc=0.01, 
                func=None):
        """
        """
        return self.network.connect(
            pre, post, transform, weight, index_pre, index_post, pstc, func)
    
    def _learn(self, pre, post, error, pstc=0.01, **kwargs):
        return self.network.learn(pre, post, error, pstc, **kwargs);
                                  
    def _make_array(self, name, neurons, array_size, dimensions=1, **kwargs):
        return self.network.make_array(name, neurons, array_size, dimensions,
                                       **kwargs)
    
    def _make_node(self, *args, **kwargs): 
        """
        XXX
        Don't you wish the docs were here?
        """
        return self.network.make_node(*args, **kwargs)
        
    def _make_subnetwork(self, name):
        raise NotImplementedError()
            
                                
