import random
import numpy as np

import neuron
from output import Output

_ARRAY_SIZE = 1

def is_ensemble(obj):
    return isinstance(obj, BaseEnsemble)

class BaseEnsemble(object):
    def __init__(self, name, dimensions):
        self.name = name
        self.dimensions = int(dimensions)

#class DirectEnsemble(Base):
#
#    def add_origin(self, name, func, dimensions):
#        if func is not None:
#            if 'initial_value' not in kwargs.keys():
#                # [func(np.zeros(self.dimensions)) for i in range(self.array_size)]
#                init = func(np.zeros(self.dimensions))
#                init = np.array([init for i in range(self.array_size)])
#                kwargs['initial_value'] = init.flatten()
#
#        if 'dt' in kwargs.keys():
#            del kwargs['dt']
#
#        self.origin[name] = origin.Origin(func=func, **kwargs) 
#
#    def add_termination(self, name, pstc, decoded_input=None, encoded_input=None):
#        """Accounts for a new termination that takes the given input
#        (a theano object) and filters it with the given pstc.
#
#        Adds its contributions to the set of decoded, encoded,
#        or learn input with the same pstc. Decoded inputs
#        are represented signals, encoded inputs are
#        decoded_output * weight matrix, learn input is
#        activities * weight_matrix.
#
#        Can only have one of decoded OR encoded OR learn input != None.
#
#        :param float pstc: post-synaptic time constant
#        :param decoded_input:
#            theano object representing the decoded output of
#            the pre population multiplied by this termination's
#            transform matrix
#        :param encoded_input:
#            theano object representing the encoded output of
#            the pre population multiplied by a connection weight matrix
#        :param learn_input:
#            theano object representing the learned output of
#            the pre population multiplied by a connection weight matrix
#        
#        """
#        # make sure one and only one of
#        # (decoded_input, encoded_input) is specified
#        if decoded_input is not None: 
#            assert (encoded_input is None)
#        elif encoded_input is not None:
#            assert (decoded_input is None) 
#        else:
#            assert False
#
#        if decoded_input: 
#            self.decoded_input[name] = filter.Filter(pstc, 
#                source=decoded_input, 
#                shape=(self.array_size, self.dimensions))
#        elif encoded_input: 
#            self.encoded_input[name] = filter.Filter(pstc, 
#                source=encoded_input, 
#                shape=(self.array_size, self.neurons))
#
#    def tick(self):
#
#        # set up matrix to store accumulated decoded input
#        X = np.zeros((self.array_size, self.dimensions))
#
#        # updates is an ordered dictionary of theano variables to update
#        for di in self.decoded_input.values(): 
#            # add its values to the total decoded input
#            X += di.value.get_value()
#
#        # if we're calculating a function on the decoded input
#        for o in self.origin.values(): 
#            if o.func is not None:  
#                val = np.float32([o.func(X[i]) for i in range(len(X))])
#                o.decoded_output.set_value(val.flatten())


class SpikingEnsemble(BaseEnsemble):
    """An ensemble is a collection of neurons representing a vector space.
    """
    
    def __init__(self, name, num_neurons, dimensions,
            neuron_model=None,
            max_rate=(200, 300),
            intercept=(-1.0, 1.0),
            radius=1.0,
            encoders=None,
            seed=None,
            decoder_noise=0.1,
            noise=None,
            ):
        """Construct an ensemble composed of the specific neuron model,
        with the specified neural parameters.

        :param tuple max_rate:
            lower and upper bounds on randomly generated
            firing rates for each neuron
        :param tuple intercept:
            lower and upper bounds on randomly generated
            x offsets for each neuron
        :param float radius:
            the range of input values (-radius:radius)
            per dimension this population is sensitive to
        :param list encoders: set of possible preferred directions
        :param int seed: seed value for random number generator
        :param string neuron_type:
            type of neuron model to use, options = {'lif'}
        :param float decoder_noise: amount of noise to assume when computing 
            decoder    
        :param string noise_type:
            the type of noise added to the input current.
            Possible options = {'uniform', 'gaussian'}.
            Default is 'uniform' to match the Nengo implementation.
        :param noise: distribution e.g. Uniform
            noise parameter for noise added to input current,
            sampled at every timestep.

        """
        BaseEnsemble.__init__(self, name, dimensions)
        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        self.radius = radius
        self.noise = noise
        self.decoder_noise = decoder_noise
        self.encoders = encoders
        if neuron_model is None:
            self.neuron_model = neuron.lif_rate.LIFRateNeuron(num_neurons)
        else:
            self.neuron_model = neuron_model
            self.neuron_model.size = num_neurons

        self.vector_inputs = []
        self.neuron_inputs = []
        self.outputs = []

        self.intercept = intercept
        self.max_rate = max_rate

    def _build(self, state, dt):

        if self.max_rate['type'].lower() == 'uniform':
            self.neuron_model.max_rates = np.random.uniform(
                size=(self.neuron_model.size, 1),
                low=self.max_rate['low'], 
                high=self.max_rate['high'])

        elif self.max_rate['type'].lower() == 'gaussian':
            self.neuron_model.max_rates = np.random.normal(
                size=(self.neuron_model.size, 1),
                loc=self.max_rate['mean'], 
                scale=self.max_rate['variance'])

        if self.intercept['type'].lower() == 'uniform':
            self.neuron_model.intercepts = np.random.uniform(
                size=(self.neuron_model.size, 1),
                low=self.max_rate['low'], 
                high=self.max_rate['high'])

        elif self.intercept['type'].lower() == 'gaussian':
            self.neuron_model.intercepts = np.random.normal(
                size=(self.neuron_model.size, 1),
                loc=self.max_rate['mean'], 
                scale=self.max_rate['variance'])

        self.neuron_model._build(state, dt)

        # compute encoders
        self.encoders = self.make_encoders(encoders=self.encoders)
        # combine encoders and gain for simplification
        self.encoders = (self.encoders.T * self.neuron_model.alpha.T).T

    def add_connection(self, connection):
        self.vector_inputs += [connection]
        
    def add_neuron_connection(self, connection):
        self.neuron_inputs += [connection]

    def add_output(self, func, dimensions, name=None):
        self.outputs += [Output(name)]
        self.output_funcs += [func]
        
        return self.outputs[-1]

    def make_encoders(self, encoders=None):
        """Generates a set of encoders.

        :param int neurons: number of neurons 
        :param int dimensions: number of dimensions
        :param theano.tensor.shared_randomstreams snrg:
            theano random number generator function
        :param list encoders:
            set of possible preferred directions of neurons

        """
        if encoders is None:
            # if no encoders specified, generate randomly
            encoders = np.random.normal(
                size=(self.neuron_model.size, self.dimensions))
        else:
            # if encoders were specified, cast list as array
            encoders = np.asarray(encoders)
            if encoders.shape[0] == self.dimensions:
                encoders = encoders.T
            # repeat array until 'encoders' is the same length
            # as number of neurons in population
            encoders = np.tile(encoders,
                (self.neuron_model.size / len(encoders) + 1)
                               )[:self.neuron_model.size, :self.dimensions]

        # normalize encoders across represented dimensions 
        norm = np.sum(encoders * encoders, axis=1).reshape(
            (self.neuron_model.size, 1))
        encoders = encoders / np.sqrt(norm) 
        
        return encoders

    def _step(self, old_state, new_state, dt):
        """Computes the new output values for this ensemble and applies
        learning to any input connections with a learning rule.

        :param float dt: the timestep of the update
        """
        
        # find the total input current to this population of neurons
        
        # apply respective biases to neurons in the population 
        J = np.zeros((self.neuron_model.size, 1))
        J += self.neuron_model.j_bias

        #add in neuron->neuron currents
        for c in self.neuron_inputs:
            # add its values directly to the input current
            J += c.get_post_input(old_state, dt)

        #add in vector->vector currents
        for c in self.vector_inputs:
            fuck = c.get_post_input(old_state, dt) 
            J += c.get_post_input(old_state, dt) * self.encoders

        # if noise has been specified for this neuron,
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise.type == 'gaussian':
                J += random.gaussian(
                    size=self.bias.shape, std=np.sqrt(self.noise/dt))
            '''elif self.noise.type == 'uniform':
                J += random.uniform(
                    size=self.bias.shape, 
                    low=-self.noise / np.sqrt(dt), 
                    high=self.noise / np.sqrt(dt))'''
        
        # pass the input current total into the neuron model
        self.spikes = self.neuron_model._step(new_state, J, dt)
    
        # update the weight matrices on learned terminations
        for c in self.vector_inputs+self.neuron_inputs:
            c.learn(dt)

        # compute the decoded origin decoded_input from the neuron output
        for i,o in enumerate(self.outputs):
            new_state[o] = self.output_funcs[i](self.neurons.output)
            
def Ensemble(*args, **kwargs):
    if kwargs.pop('mode', 'spiking') == 'spiking':
        return SpikingEnsemble(*args, **kwargs)
    else:
#        return DirectEnsemble(*args, **kwargs)
        raise NotImplementedError()
