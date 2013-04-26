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
        self.eval_points = None
        if neuron_model is None:
            self.neuron_model = neuron.lif_rate.LIFRateNeuron(num_neurons)
        else:
            self.neuron_model = neuron_model
            self.neuron_model.size = num_neurons

        self.vector_inputs = []
        self.neuron_inputs = []

        # Track the outputs from this ensemble, 
        self.outputs = []
        # the function they should be computing, 
        self.output_funcs = []
        # and the decoders that approximate this.
        self.decoders = []

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
                low=self.intercept['low'], 
                high=self.intercept['high'])
        elif self.intercept['type'].lower() == 'gaussian':
            self.neuron_model.intercepts = np.random.normal(
                size=(self.neuron_model.size, 1),
                loc=self.intercept['mean'], 
                scale=self.intercept['variance'])

        self.neuron_model._build(state, dt)

        # compute encoders
        self.encoders = self.make_encoders(encoders=self.encoders)

        # compute the decoded origin decoded_input from the neuron output
        for i,o in enumerate(self.outputs):
            state[o] = np.zeros(o.dimensions)
            self.decoders += [
                self.compute_decoders(func=self.output_funcs[i], 
                dt=dt, eval_points=self.eval_points) ]

    def add_connection(self, connection):
        self.vector_inputs += [connection]
        
    def add_neuron_connection(self, connection):
        self.neuron_inputs += [connection]

    def add_output(self, func):
        self.outputs += [ Output(name=func.__name__, 
            dimensions=func(np.zeros(self.dimensions)).shape[0]) ]
        self.output_funcs += [func]
        
        return self.outputs[-1]

    def compute_decoders(self, func, dt, eval_points=None):     
        """Compute decoding weights.

        Calculate the scaling values to apply to the output
        of each of the neurons in the attached population
        such that the weighted summation of their output
        generates the desired decoded output.

        Decoder values computed as D = (A'A)^-1 A'X_f
        where A is the matrix of activity values of each 
        neuron over sampled X values, and X_f is the vector
        of desired f(x) values across sampled points.

        :param function func: function to compute with this origin
        :param float dt: timestep for simulating to get A matrix
        :param list eval_points:
            specific set of points to optimize decoders over 
        """

        if eval_points == None:  
            # generate sample points from state space randomly
            # to minimize decoder error over in decoder calculation
            self.num_samples = 200 ##################################################33should be 500!!!!
            samples = np.random.uniform(
                size=(self.num_samples, self.dimensions),
                low=-1, high=1)

            # normalize magnitude of sampled points to be of unit length
            norm = np.sum(samples * samples, axis=1, dtype='float')[:, np.newaxis]
            samples = samples / np.sqrt(norm)

            # generate magnitudes for vectors from uniform distribution
            scale = (np.random.uniform(size=(self.num_samples,))
                     ** (1.0 / self.dimensions))

            # scale sample points
            samples = samples.T * scale 

            eval_points = samples

        else:
            # otherwise reset num_samples, and 
            # make sure eval_points is in the right form
            # (rows are input dimensions, columns different samples)
            eval_points = np.array(eval_points)
            if len(eval_points.shape) == 1:
                eval_points.shape = [1, eval_points.shape[0]]
            self.num_samples = eval_points.shape[1]

            if eval_points.shape[0] != self.dimensions: 
                raise Exception("Evaluation points must be of the form: " + 
                    "[dimensions x num_samples]")

        # compute the target_values at the sampled points 
        if func is None:
            # if no function provided, use identity function as default
            target_values = eval_points 
        else:
            # otherwise calculate target_values using provided function
            
            # scale all our sample points by ensemble radius,
            # calculate function value, then scale back to unit length

            # this ensures that we accurately capture the shape of the
            # function when the radius is > 1 (think for example func=x**2)
            target_values = \
                (np.array(
                    [func(s * self.radius) for s in eval_points.T]
                    ) / self.radius )
            if len(target_values.shape) < 2:
                target_values.shape = target_values.shape[0], 1
            target_values = target_values.T
        
        # compute the input current for every neuron and every sample point
        J = np.dot(self.encoders, eval_points)

        A = np.zeros((self.neuron_model.size, self.num_samples))
        for i in range(self.num_samples):

            # simulate neurons for .25 seconds to get startup 
            # transient out of the way
            state = {}
            for t in range(int(.05/dt)): 
                self.neuron_model._step(state, J[:,i], dt)

            # run the neuron model for 1 second,
            # accumulating spikes to get a spike rate
            num_time_samples = int(.1/dt)
            firing_rates = np.zeros((self.neuron_model.size, num_time_samples))
            for t in range(num_time_samples): 
                self.neuron_model._step(state, J[:,i], dt)
                firing_rates[:,t] = state[self.neuron_model.output]

            # TODO: np.mean instead?
            A[:,i] = np.sum(firing_rates, axis=1) / num_time_samples
            self.neuron_model._reset(state)

        # add noise to elements of A
        # std_dev = max firing rate of population * .1
        noise = 0.1 # from Nengo
        A += noise * np.random.normal(
            size=(self.neuron_model.size, self.num_samples), 
            scale=(self.neuron_model.max_rates.max()))

        # compute Gamma and Upsilon
        G = np.dot(A, A.T) # correlation matrix
        
        # eigh for symmetric matrices, returns
        # evalues w and normalized evectors v
        w, v = np.linalg.eigh(G)

        dnoise = self.decoder_noise * self.decoder_noise

        # formerly 0.1 * 0.1 * max(w), set threshold
        limit = dnoise * max(w) 
        for i in range(len(w)):
            if w[i] < limit:
                # if < limit set eval = 0
                w[i] = 0
            else:
                # prep for upcoming Ginv calculation
                w[i] = 1.0 / w[i]
        # w[:, np.newaxis] gives transpose of vector,
        # np.multiply is very fast element-wise multiplication
        Ginv = np.dot(v, np.multiply(w[:, np.newaxis], v.T)) 
        
        U = np.dot(A, target_values.T)
        
        # compute decoders - least squares method 
        decoders = np.dot(Ginv, U)

        return decoders

    def get(self, name):
        search = [x for x in self.outputs if x.name == name] + \
                [self for x in self.vector_inputs+self.neuron_inputs if x.post == self.name + ":" + name]
        if len(search) > 1:
            print "Warning, found more than one object with same name"
        if len(search) == 0:
            print name + " not found in ensemble.get"
            return self
        return search[0]

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
        J = np.zeros(self.neuron_model.size)

        '''#add in neuron->neuron currents
        for c in self.neuron_inputs:
            # add its values directly to the input current
            J = J + c.get_post_input(old_state, dt)'''

        #add in vector->vector currents
        for c in self.vector_inputs:
            fuck = c.get_post_input(old_state, dt)
            J = J + np.dot( self.encoders, 
                c.get_post_input(old_state, dt))

        '''# if noise has been specified for this neuron,
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise.type == 'gaussian':
                J += np.random.normal(
                    size=self.bias.shape, std=np.sqrt(self.noise/dt))
            elif self.noise.type == 'uniform':
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
            new_state[o] = np.dot(self.spikes, self.decoders[i]).flatten()

def Ensemble(*args, **kwargs):
    if kwargs.pop('mode', 'spiking') == 'spiking':
        return SpikingEnsemble(*args, **kwargs)
    else:
#        return DirectEnsemble(*args, **kwargs)
        raise NotImplementedError()
