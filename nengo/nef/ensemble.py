try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict

import numpy as np

from . import neuron
from . import ensemble_origin
from . import origin
from . import cache
from . import filter
from .hPES_termination import hPESTermination

_ARRAY_SIZE = 1

class Uniform(object):
    def __init__(self, low, high):
        self.type = 'uniform'
        self.low = low
        self.high = high


class Gaussian(object):
    def __init__(self, low, high):
        self.type = 'gaussian'
        self.low = low
        self.high = high


class Base(object):
    def __init__(self, dimensions, array_size=_ARRAY_SIZE):
        self.dimensions = dimensions
        self.array_size = array_size

        self.origin = OrderedDict()

        self.decoded_input = OrderedDict()

        # set up a dictionary for encoded_input connections
        self.encoded_input = {}

        # list of learned terminations on ensemble
        self.learned_terminations = []


class DirectEnsemble(Base):

    def add_origin(self, name, func, dimensions):
        if func is not None:
            if 'initial_value' not in kwargs.keys():
                # [func(np.zeros(self.dimensions)) for i in range(self.array_size)]
                init = func(np.zeros(self.dimensions))
                init = np.array([init for i in range(self.array_size)])
                kwargs['initial_value'] = init.flatten()

        if 'dt' in kwargs.keys():
            del kwargs['dt']

        self.origin[name] = origin.Origin(func=func, **kwargs) 

    def add_termination(self, name, pstc, decoded_input=None, encoded_input=None):
        """Accounts for a new termination that takes the given input
        (a theano object) and filters it with the given pstc.

        Adds its contributions to the set of decoded, encoded,
        or learn input with the same pstc. Decoded inputs
        are represented signals, encoded inputs are
        decoded_output * weight matrix, learn input is
        activities * weight_matrix.

        Can only have one of decoded OR encoded OR learn input != None.

        :param float pstc: post-synaptic time constant
        :param decoded_input:
            theano object representing the decoded output of
            the pre population multiplied by this termination's
            transform matrix
        :param encoded_input:
            theano object representing the encoded output of
            the pre population multiplied by a connection weight matrix
        :param learn_input:
            theano object representing the learned output of
            the pre population multiplied by a connection weight matrix
        
        """
        # make sure one and only one of
        # (decoded_input, encoded_input) is specified
        if decoded_input is not None: 
            assert (encoded_input is None)
        elif encoded_input is not None:
            assert (decoded_input is None) 
        else:
            assert False

        if decoded_input: 
            self.decoded_input[name] = filter.Filter(pstc, 
                source=decoded_input, 
                shape=(self.array_size, self.dimensions))
        elif encoded_input: 
            self.encoded_input[name] = filter.Filter(pstc, 
                source=encoded_input, 
                shape=(self.array_size, len(self.neurons)))

    def tick(self):

        # set up matrix to store accumulated decoded input
        X = np.zeros((self.array_size, self.dimensions))

        # updates is an ordered dictionary of theano variables to update
        for di in self.decoded_input.values(): 
            # add its values to the total decoded input
            X += di.value.get_value()

        # if we're calculating a function on the decoded input
        for o in self.origin.values(): 
            if o.func is not None:  
                val = np.float32([o.func(X[i]) for i in range(len(X))])
                o.decoded_output.set_value(val.flatten())


class SpikingEnsemble(Base):
    """An ensemble is a collection of neurons representing a vector space.
    """
    
    def __init__(self, neurons, dimensions, array_size=_ARRAY_SIZE,
            max_rate=(200, 300), intercept=(-1.0, 1.0),
            radius=1.0,
            encoders=None,
            seed=None,
            array_size=_ARRAY_SIZE,
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
        :param int array_size: number of sub-populations for network arrays
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
        Base.__init__(self, dimensions, array_size)
        if seed is None:
            seed = np.random.randint(1000)
        self.neurons = neurons
        if len(neurons) % array_size:
            raise ValueError('array_size must divide population size',
                    (len(neurons), array_size))
        self.seed = seed
        self.radius = radius
        self.noise = noise
        self.decoder_noise = decoder_noise

        # make sure intercept is the right shape
        if isinstance(intercept, (int,float)):
            intercept = [intercept, 1]
        elif len(intercept) == 1:
            intercept.append(1) 

        # compute alpha and bias
        self.rng = np.random.RandomState(seed=seed)
        self.max_rate = max_rate
        max_rates = self.rng.uniform(
            size=(self.array_size, len(self.neurons)),
            low=max_rate[0], high=max_rate[1])  
        threshold = self.rng.uniform(
            size=(self.array_size, len(self.neurons)),
            low=intercept[0], high=intercept[1])
        alpha, self.bias = self.neurons.make_alpha_bias(max_rates, threshold)

        # force to 32 bit for consistency / speed
        self.bias = self.bias.astype('float32')
                
        # compute encoders
        self.encoders = self.make_encoders(encoders=encoders)
        # combine encoders and gain for simplification
        self.encoders = (self.encoders.T * alpha.T).T

    def add_termination(self, name, pstc, decoded_input=None, encoded_input=None):
        """Accounts for a new termination that takes the given input
        (a theano object) and filters it with the given pstc.

        Adds its contributions to the set of decoded, encoded,
        or learn input with the same pstc. Decoded inputs
        are represented signals, encoded inputs are
        decoded_output * weight matrix, learn input is
        activities * weight_matrix.

        Can only have one of decoded OR encoded OR learn input != None.

        :param float pstc: post-synaptic time constant
        :param decoded_input:
            theano object representing the decoded output of
            the pre population multiplied by this termination's
            transform matrix
        :param encoded_input:
            theano object representing the encoded output of
            the pre population multiplied by a connection weight matrix
        :param learn_input:
            theano object representing the learned output of
            the pre population multiplied by a connection weight matrix
        
        """
        # make sure one and only one of
        # (decoded_input, encoded_input) is specified
        if decoded_input is not None: 
            assert (encoded_input is None)
        elif encoded_input is not None:
            assert (decoded_input is None) 
        else:
            assert False

        if decoded_input: 
            # rescale decoded_input by this neuron's radius
            self.decoded_input[name] = filter.Filter(pstc, 
                source=np.true_div(decoded_input, self.radius), 
                shape=(self.array_size, self.dimensions))
        elif encoded_input: 
            self.encoded_input[name] = filter.Filter(pstc, 
                source=encoded_input, 
                shape=(self.array_size, len(self.neurons)))

    def add_learned_termination(self, name, pre, error, pstc, 
                                learned_termination_class=hPESTermination,
                                **kwargs):
        """Adds a learned termination to the ensemble.

        Input added to encoded_input, and a learned_termination object
        is created to keep track of the pre and post
        (self) spike times, and adjust the weight matrix according
        to the specified learning rule.

        :param Ensemble pre: the pre-synaptic population
        :param Ensemble error: the Origin that provides the error signal
        :param float pstc:
        :param learned_termination_class:
        """
        #TODO: is there ever a case we wouldn't want this?
        assert error.dimensions == self.dimensions * self.array_size

        # generate an initial weight matrix if none provided,
        # random numbers between -.001 and .001
        if 'weight_matrix' not in kwargs.keys():
            weight_matrix = np.random.uniform(
                size=(self.array_size * pre.array_size,
                      len(self.neurons), len(pre.neurons)),
                low=-.001, high=.001)
            kwargs['weight_matrix'] = weight_matrix
        else:
            # make sure it's an np.array
            #TODO: error checking to make sure it's the right size
            kwargs['weight_matrix'] = np.array(kwargs['weight_matrix']) 

        learned_term = learned_termination_class(
            pre=pre, post=self, error=error, **kwargs)

        learn_projections = [np.dot(
            pre.neurons.output[learned_term.pre_index(i)],  
            learned_term.weight_matrix[i % self.array_size]) 
            for i in range(self.array_size * pre.array_size)]

        # now want to sum all the output to each of the post ensembles 
        # going to reshape and sum along the 0 axis
        learn_output = np.sum( 
            np.reshape(learn_projections, 
            (pre.array_size, self.array_size, len(self.neurons))), axis=0)
        # reshape to make it (array_size x len(self.neurons))
        learn_output = np.reshape(learn_output, 
            (self.array_size, len(self.neurons)))

        # the input_current from this connection during simulation
        self.add_termination(name=name, pstc=pstc, encoded_input=learn_output)
        self.learned_terminations.append(learned_term)
        return learned_term

    def add_origin(self, name, func, **kwargs):
        """Create a new origin to perform a given function
        on the represented signal.

        :param string name: name of origin
        :param function func:
            desired transformation to perform over represented signal
        :param list eval_points:
            specific set of points to optimize decoders over for this origin
        """

        if 'eval_points' not in kwargs.keys():
            kwargs['eval_points'] = self.eval_points
        self.origin[name] = ensemble_origin.EnsembleOrigin(
            ensemble=self, func=func, **kwargs)


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
            encoders = self.rng.normal(
                size=(self.array_size, len(self.neurons), self.dimensions))
            assert encoders.ndim == 3
        else:
            # if encoders were specified, cast list as array
            encoders = np.array(encoders).T
            # repeat array until 'encoders' is the same length
            # as number of neurons in population
            encoders = np.tile(encoders,
                (len(self.neurons) / len(encoders) + 1)
                               ).T[:len(self.neurons), :self.dimensions]
            encoders = np.tile(encoders, (self.array_size, 1, 1))

            assert encoders.ndim == 3
        # normalize encoders across represented dimensions 
        print encoders.shape
        norm = np.sum(encoders * encoders, axis=2)[:, :, None]
        encoders = encoders / np.sqrt(norm)        

        return encoders


    def tick(self, dt):
        """Compute the set of theano updates needed for this ensemble.

        Returns a dictionary with new neuron state,
        termination, and origin values.

        :param float dt: the timestep of the update
        """
        
        ### find the total input current to this population of neurons

        # set up matrix to store accumulated decoded input
        X = np.zeros((self.array_size, self.dimensions))
    
        # apply respective biases to neurons in the population 
        J = np.array(self.bias)

        for ei in self.encoded_input.values():
            # add its values directly to the input current
            J += ei.value
            updates.update(ei.update(dt))

        # only do this if there is decoded_input
        if len(self.decoded_input) > 0:
            # add to input current for each neuron as
            # represented input signal x preferred direction
            J = [J[i] + np.dot(self.shared_encoders[i], X[i].T)
                 for i in range(self.array_size)]

        # if noise has been specified for this neuron,
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise.type == 'gaussian':
                J += self.rng.normal(
                    size=self.bias.shape, std=np.sqrt(self.noise/dt))
            elif self.noise.type == 'uniform':
                J += self.rng.uniform(
                    size=self.bias.shape, 
                    low=-self.noise / np.sqrt(dt), 
                    high=self.noise / np.sqrt(dt))

        # pass that total into the neuron model to produce
        # the main theano computation
        updates.update(self.neurons.update(J, dt))
    
        for l in self.learned_terminations:
            # also update the weight matrices on learned terminations
            updates.update(l.update(dt))

        # and compute the decoded origin decoded_input from the neuron output
        for o in self.origin.values():
            updates.update(o.update(dt, updates[self.neurons.output]))

        return updates


def Ensemble(*args, **kwargs):
    if kwargs.pop('mode', 'spiking') == 'spiking':
        return SpikingEnsemble(*args, **kwargs)
    else:
        return DirectEnsemble(*args, **kwargs)

