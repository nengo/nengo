"""
old_api.py: adapter for nengo_theano and [Jython] nengo-1.4

The purpose of this emulation layer is to run scripts and tests that were
written for nengo-1.4 and theano_nengo. Authors are encouraged to use
the "api.py" file instead of this file for their current work.

"""

import random
import numpy as np

from simulator_objects import ShapeMismatch
from simulator_objects import SimModel
from nonlinear import LIF
from nonlinear import LIFRate
from nonlinear import Direct

neuron_type_registry = {
    'lif': LIF,
    'lif-rate': LIFRate,
        }

from . import simulator


def compute_transform(dim_pre, dim_post, array_size_post, array_size_pre,
        weight=1, index_pre=None, index_post=None, transform=None):
    """Helper function used by :func:`nef.Network.connect()` to create
    the `dim_post` by `dim_pre` transform matrix.

    Values are either 0 or *weight*. *index_pre* and *index_post*
    are used to determine which values are non-zero, and indicate
    which dimensions of the pre-synaptic ensemble should be routed
    to which dimensions of the post-synaptic ensemble.

    :param int dim_pre: first dimension of transform matrix
    :param int dim_post: second dimension of transform matrix
    :param int array_size: size of the network array
    :param float weight: the non-zero value to put into the matrix
    :param index_pre: the indexes of the pre-synaptic dimensions to use
    :type index_pre: list of integers or a single integer
    :param index_post:
        the indexes of the post-synaptic dimensions to use
    :type index_post: list of integers or a single integer
    :returns:
        a two-dimensional transform matrix performing
        the requested routing

    """

    all_pre = dim_pre * array_size_pre

    if transform is None:
        # create a matrix of zeros
        transform = [[0] * all_pre for i in range(dim_post * array_size_post)]

        # default index_pre/post lists set up *weight* value
        # on diagonal of transform

        # if dim_post * array_size_post != all_pre,
        # then values wrap around when edge hit
        if index_pre is None:
            index_pre = range(all_pre)
        elif isinstance(index_pre, int):
            index_pre = [index_pre]
        if index_post is None:
            index_post = range(dim_post * array_size_post)
        elif isinstance(index_post, int):
            index_post = [index_post]

        for i in range(max(len(index_pre), len(index_post))):
            pre = index_pre[i % len(index_pre)]
            post = index_post[i % len(index_post)]
            transform[post][pre] = weight

    transform = np.asarray(transform)

    # reformulate to account for post.array_size_post
    if transform.shape == (dim_post * array_size_post, all_pre):
        rval = np.zeros((array_size_pre, dim_pre, array_size_post, dim_post))
        for i in range(array_size_post):
            for j in range(dim_post):
                rval[:, :, i, j] = transform[i * dim_post + j].reshape(
                        array_size_pre, dim_pre)

        transform = rval
    else:
        raise NotImplementedError()

    rval = np.asarray(transform)
    return rval


def sample_unit_signal(dimensions, num_samples, rng):
    """Generate sample points uniformly distributed within the sphere.

    Returns float array of sample points: dimensions x num_samples

    """
    samples = rng.randn(num_samples, dimensions)

    # normalize magnitude of sampled points to be of unit length
    norm = np.sum(samples * samples, axis=1)
    samples /= np.sqrt(norm)[:, None]

    # generate magnitudes for vectors from uniform distribution
    scale = rng.rand(num_samples, 1) ** (1.0 / dimensions)

    # scale sample points
    samples *= scale

    return samples.T


def filter_coefs(pstc, dt):
    """
    Use like: fcoef, tcoef = filter_coefs(pstc=pstc, dt=dt)
        transform(tcoef, a, b)
        filter(fcoef, b, b)
    """
    pstc = max(pstc, dt)
    decay = np.exp(-dt / pstc)
    return decay, (1.0 - decay)


# -- James and Terry arrived at this by eyeballing some graphs.
#    Not clear if this should be a constant at all, it
#    may depend on fn being estimated, number of neurons, etc...
DEFAULT_RCOND=0.01

class EnsembleOrigin(object):
    def __init__(self, ensemble, name, func=None,
            pts_slice=slice(None, None, None),
            rcond=DEFAULT_RCOND,
            ):
        """The output from a population of neurons (ensemble),
        performing a transformation (func) on the represented value.

        :param Ensemble ensemble:
            the Ensemble to which this origin is attached
        :param function func:
            the transformation to perform to the ensemble's
            represented values to get the output value
        """
        eval_points = ensemble.babbling_signal

        # compute the targets at the sampled points
        if func is None:
            # if no function provided, use identity function as default
            targets = eval_points.T
        else:
            # otherwise calculate targets using provided function

            # scale all our sample points by ensemble radius,
            # calculate function value, then scale back to unit length

            # this ensures that we accurately capture the shape of the
            # function when the radius is > 1 (think for example func=x**2)
            targets = np.array([func(s) for s in eval_points.T])
            if len(targets.shape) < 2:
                targets.shape = targets.shape[0], 1

        n, = targets.shape[1:]
        dt = ensemble.model.dt
        self.sigs = []
        self.decoders = []
        self.transforms = []

        for ii in range(ensemble.array_size):
            # -- N.B. this is only accurate for models firing well
            #    under the simulator's dt.
            A = ensemble.neurons[ii].babbling_rate * dt
            b = targets
            weights, res, rank, s = np.linalg.lstsq(A, b, rcond=rcond)

            sig = ensemble.model.signal(n=n, name='%s[%i]' % (name, ii))
            decoder = ensemble.model.decoder(
                sig=sig,
                pop=ensemble.neurons[ii],
                weights=weights.T)

            # set up self.sig as an unfiltered signal
            transform = ensemble.model.transform(1.0, sig, sig)

            self.sigs.append(sig)
            self.decoders.append(decoder)
            self.transforms.append(transform)


class Ensemble:
    """An ensemble is a collection of neurons representing a vector space.
    """
    def __init__(self, model, name, neurons, dimensions, dt, tau_ref=0.002, tau_rc=0.02,
                 max_rate=(200, 300), intercept=(-1.0, 1.0), radius=1.0,
                 encoders=None, seed=None, neuron_type='lif',
                 array_size=1, eval_points=None, decoder_noise=0.1,
                 noise_type='uniform', noise=None, mode='spiking'):
        """Construct an ensemble composed of the specific neuron model,
        with the specified neural parameters.

        :param int neurons: number of neurons in this population
        :param int dimensions:
            number of dimensions in the vector space
            that these neurons represent
        :param float tau_ref: length of refractory period
        :param float tau_rc:
            RC constant; approximately how long until 2/3
            of the threshold voltage is accumulated
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
        :param list eval_points:
            specific set of points to optimize decoders over by default
        :param float decoder_noise: amount of noise to assume when computing
            decoder
        :param string noise_type:
            the type of noise added to the input current.
            Possible options = {'uniform', 'gaussian'}.
            Default is 'uniform' to match the Nengo implementation.
        :param float noise:
            noise parameter for noise added to input current,
            sampled at every timestep.
            If noise_type = uniform, this is the lower and upper
            bound on the distribution.
            If noise_type = gaussian, this is the variance.

        """
        if seed is None:
            seed = np.random.randint(1000)
        self.n_neurons = neurons
        del neurons  # neurons usually means the nonlinearities
        n_neurons = self.n_neurons
        self.dimensions = dimensions
        self.array_size = array_size
        self.radius = radius
        self.noise = noise
        self.noise_type = noise_type
        self.decoder_noise = decoder_noise
        self.mode = mode
        self.model = model
        self.name = name

        # make sure that eval_points is the right shape
        if eval_points is not None:
            eval_points = np.array(eval_points)
            if len(eval_points.shape) == 1:
                eval_points.shape = [1, eval_points.shape[0]]
        self.eval_points = eval_points

        # make sure intercept is the right shape
        if isinstance(intercept, (int,float)):
            intercept = [intercept, 1]
        elif len(intercept) == 1:
            intercept.append(1)

        # make dictionary for origins
        self.origin = {}
        # set up a dictionary for decoded_input
        self.decoded_input = {}

        self.input_signals = [
            model.signal(
                n=dimensions,
                name='%s.input_signals[%i]' % (name, ii))
            for ii in range(array_size)]

        # if we're creating a spiking ensemble
        if self.mode == 'spiking':

            self.rng = np.random.RandomState(seed)
            self.max_rate = max_rate

            self._make_encoders(encoders)
            self.encoders /= radius
            self.babbling_signal = sample_unit_signal(
                    self.dimensions, 500, self.rng) * radius

            self.neurons = []
            for ii in range(array_size):
                neurons_ii = self.model.nonlinearity(
                        # TODO: handle different neuron types,
                        LIF(n_neurons, tau_rc=tau_rc, tau_ref=tau_ref,
                            name=name + '[%i]' % ii),
                    )
                self.neurons.append(neurons_ii)
                max_rates = self.rng.uniform(
                    size=self.n_neurons,
                    low=max_rate[0], high=max_rate[1])
                threshold = self.rng.uniform(
                    size=self.n_neurons,
                    low=intercept[0], high=intercept[1])
                neurons_ii.set_gain_bias(max_rates, threshold)

                # pre-multiply encoder weights by gain
                self.encoders[ii] *= neurons_ii.gain[:, None]

                # -- alias self.encoders to the matrices
                # in the model encoders (objects)
                self.model.encoder(
                    self.input_signals[ii],
                    neurons_ii,
                    weights=self.encoders[ii])
                neurons_ii.babbling_rate = neurons_ii.rates(
                        np.dot(
                            self.encoders[ii],
                            self.babbling_signal).T)

            # set up a dictionary for encoded_input connections
            self.encoded_input = {}
            # list of learned terminations on ensemble
            self.learned_terminations = []

            # make default origin
            self.add_origin('X')

        elif self.mode == 'direct':
            # make default origin
            self.add_origin('X',
                            dimensions=self.dimensions*self.array_size)
            # reset n_neurons to 0
            self.n_neurons = 0

    def add_termination(self, name, pstc,
                        decoded_input=None, encoded_input=None):
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
        raise NotImplementedError()
        # make sure one and only one of
        # (decoded_input, encoded_input) is specified
        if decoded_input is not None: assert (encoded_input is None)
        elif encoded_input is not None: assert (decoded_input is None)
        else: assert False

        if decoded_input:
            if self.mode is not 'direct':
                # rescale decoded_input by this neuron's radius
                source = TT.true_div(decoded_input, self.radius)
            # ignore radius in direct mode
            else: source = decoded_input
            name = self.get_unique_name(name, self.decoded_input)
            self.decoded_input[name] = filter.Filter(
                name=name, pstc=pstc, source=source,
                shape=(self.array_size, self.dimensions))
        elif encoded_input:
            name = self.get_unique_name(name, self.encoded_input)
            self.encoded_input[name] = filter.Filter(
                name=name, pstc=pstc, source=encoded_input,
                shape=(self.array_size, self.n_neurons))

    def add_learned_termination(self, name, pre, error, pstc,
                                learned_termination_class=None,
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
        raise NotImplementedError()
        #TODO: is there ever a case we wouldn't want this?
        assert error.dimensions == self.dimensions * self.array_size

        # generate an initial weight matrix if none provided,
        # random numbers between -.001 and .001
        if 'weight_matrix' not in kwargs.keys():
            # XXX use self.rng
            weight_matrix = np.random.uniform(
                size=(self.array_size * pre.array_size,
                      self.n_neurons, pre.n_neurons),
                low=-.001, high=.001)
            kwargs['weight_matrix'] = weight_matrix
        else:
            # make sure it's an np.array
            #TODO: error checking to make sure it's the right size
            kwargs['weight_matrix'] = np.array(kwargs['weight_matrix'])

        learned_term = learned_termination_class(
            pre=pre, post=self, error=error, **kwargs)

        learn_projections = [TT.dot(
            pre.neurons.output[learned_term.pre_index(i)],
            learned_term.weight_matrix[i % self.array_size])
            for i in range(self.array_size * pre.array_size)]

        # now want to sum all the output to each of the post ensembles
        # going to reshape and sum along the 0 axis
        learn_output = TT.sum(
            TT.reshape(learn_projections,
            (pre.array_size, self.array_size, self.n_neurons)), axis=0)
        # reshape to make it (array_size x n_neurons)
        learn_output = TT.reshape(learn_output,
            (self.array_size, self.n_neurons))

        # the input_current from this connection during simulation
        self.add_termination(name=name, pstc=pstc, encoded_input=learn_output)
        self.learned_terminations.append(learned_term)
        return learned_term

    def add_origin(self, name, func=None, **kwargs):
        """Create a new origin to perform a given function
        on the represented signal.

        :param string name: name of origin
        :param function func:
            desired transformation to perform over represented signal
        :param list eval_points:
            specific set of points to optimize decoders over for this origin
        """

        # if we're in spiking mode create an ensemble_origin with decoders
        # and the whole shebang for interpreting the neural activity
        if self.mode == 'spiking':
            self.origin[name] = EnsembleOrigin(
                ensemble=self,
                func=func,
                name='%s.%s' % (self.name, name),
                **kwargs)

        # if we're in direct mode then this population is just directly
        # performing the specified function, use a basic origin
        elif self.mode == 'direct':
            raise NotImplementedError()
            if func is not None:
                if 'initial_value' not in kwargs.keys():
                    # [func(np.zeros(self.dimensions)) for i in range(self.array_size)]
                    init = func(np.zeros(self.dimensions))
                    init = np.array([init for i in range(self.array_size)])
                    kwargs['initial_value'] = init.flatten()

            if kwargs.has_key('dt'): del kwargs['dt']
            self.origin[name] = origin.Origin(func=func, **kwargs)

    def get_unique_name(self, name, dic):
        """A helper function that runs through a dictionary
        and checks for the key name, adds a digit to the end
        until a unique key has been created.

        :param string name: desired key name
        :param dict dic: the dictionary to search through
        :returns string: a unique key name for dic
        """
        i = 0
        while dic.has_key(name + '_' + str(i)):
            i += 1

        return name + '_' + str(i)

    def _make_encoders(self, encoders):
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
            encoders = self.rng.randn(
                self.array_size, self.n_neurons, self.dimensions)
        else:
            # if encoders were specified, cast list as array
            encoders = np.array(encoders).T
            # repeat array until 'encoders' is the same length
            # as number of neurons in population
            encoders = np.tile(encoders,
                (self.n_neurons / len(encoders) + 1)
                               ).T[:self.n_neurons, :self.dimensions]
            encoders = np.tile(encoders, (self.array_size, 1, 1))

        # normalize encoders across represented dimensions
        norm = np.sum(encoders * encoders, axis=2)[:, :, None]
        self.encoders = encoders / np.sqrt(norm)

    def theano_tick(self):
        if self.mode == 'direct':
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

    def update(self, dt):
        """Compute the set of theano updates needed for this ensemble.

        Returns a dictionary with new neuron state,
        termination, and origin values.

        :param float dt: the timestep of the update
        """

        ### find the total input current to this population of neurons

        # set up matrix to store accumulated decoded input
        X = None
        # updates is an ordered dictionary of theano variables to update
        updates = OrderedDict()

        for ii, di in enumerate(self.decoded_input.values()):
            # add its values to the total decoded input
            if ii == 0:
                X = di.value
            else:
                X += di.value

            updates.update(di.update(dt))

        # if we're in spiking mode, then look at the input current and
        # calculate new neuron activities for output
        if self.mode == 'spiking':

            # apply respective biases to neurons in the population
            J = TT.as_tensor_variable(np.array(self.bias))

            for ei in self.encoded_input.values():
                # add its values directly to the input current
                J += (ei.value.T * self.alpha.T).T
                updates.update(ei.update(dt))

            # only do this if there is decoded_input
            if X is not None:
                # add to input current for each neuron as
                # represented input signal x preferred direction
                J = map_gemv(1.0, self.shared_encoders, X, 1.0, J)

            # if noise has been specified for this neuron,
            if self.noise:
                # generate random noise values, one for each input_current element,
                # with standard deviation = sqrt(self.noise=std**2)
                # When simulating white noise, the noise process must be scaled by
                # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
                if self.noise_type.lower() == 'gaussian':
                    J += self.srng.normal(
                        size=self.bias.shape, std=np.sqrt(self.noise/dt))
                elif self.noise_type.lower() == 'uniform':
                    J += self.srng.uniform(
                        size=self.bias.shape,
                        low=-self.noise/np.sqrt(dt),
                        high=self.noise/np.sqrt(dt))

            # pass that total into the neuron model to produce
            # the main theano computation
            updates.update(self.neurons.update(J, dt))

            for l in self.learned_terminations:
                # also update the weight matrices on learned terminations
                updates.update(l.update(dt))

            # and compute the decoded origin decoded_input from the neuron output
            for o in self.origin.values():
                updates.update(o.update(dt, updates[self.neurons.output]))

        if self.mode == 'direct':

            # if we're in direct mode then just directly pass the decoded_input
            # to the origins for decoded_output
            for o in self.origin.values():
                if o.func is None:
                    if len(self.decoded_input) > 0:
                        updates.update(OrderedDict({o.decoded_output:
                            TT.flatten(X).astype('float32')}))
        return updates


class Probe(object):
    def __init__(self, probe, net):
        self.probe = probe
        self.net = net

    def get_data(self):
        sim = self.net.sim
        lst = sim.probe_data(self.probe)
        rval = np.asarray(lst).reshape(len(lst), -1)
        return rval


class Network(object):
    def __init__(self, name,
            seed=None,
            fixed_seed=None,
            dt=0.001,
            Simulator=simulator.Simulator):
        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)
        self.fixed_seed = fixed_seed
        self.model = SimModel(dt)
        self.ensembles = {}
        self.inputs = {}

        self.steps = self.model.signal(name='steps')
        self.simtime = self.model.signal(name='simtime')
        self.one = self.model.signal(value=[1.0], name='one')

        # -- steps counts by 1.0
        self.model.filter(1.0, self.one, self.steps)
        self.model.filter(1.0, self.steps, self.steps)

        # simtime <- dt * steps
        self.model.filter(dt, self.one, self.simtime)
        self.model.filter(dt, self.steps, self.simtime)

        self.Simulator = Simulator

    @property
    def dt(self):
        return self.model.dt

    def make_input(self, name, value):
        if callable(value):
            rval = self.model.signal()
            pop = self.model.nonlinearity(
                Direct(n_in=1, n_out=1, fn=value))
            self.model.encoder(self.simtime, pop, weights=np.asarray([[1]]))
            self.inputs[name] = pop.output_signal
            # TODO: add this to simulator_objects
            pop.input_signal.name = name + '.input'
            pop.bias_signal.name = name + '.bias'
            pop.output_signal.name = name + '.output'

            # move from signals_tmp -> signals
            self.model.transform(1.0,
                                 pop.output_signal,
                                 pop.output_signal)
        else:
            value = np.asarray(value, dtype='float')
            N, = value.shape
            rval = self.model.signal(n=N, value=value, name=name)
            self.inputs[name] = rval
        return rval

    def make_array(self, name, neurons, array_size, dimensions=1, **kwargs):
        """Generate a network array specifically.

        This function is depricated; use for legacy code
        or non-theano API compatibility.
        """
        return self.make(
            name=name, neurons=neurons, dimensions=dimensions,
            array_size=array_size, **kwargs)

    def make(self, name, *args, **kwargs):
        if 'seed' not in kwargs.keys():
            if self.fixed_seed is not None:
                kwargs['seed'] = self.fixed_seed
            else:
                # if no seed provided, get one randomly from the rng
                kwargs['seed'] = self.random.randrange(0x7fffffff)

        kwargs['dt'] = self.dt
        rval = Ensemble(self.model, name, *args, **kwargs)

        self.ensembles[name] = rval
        return rval

    def connect(self, name1, name2,
                func=None,
                pstc=0.01,
                **kwargs):
        if name1 in self.ensembles:
            src = self.ensembles[name1]
            dst = self.ensembles[name2]
            if func is None:
                oname = 'X'
            else:
                oname = func.__name__

            if oname not in src.origin:
                src.add_origin(oname, func)

            decoded_origin = src.origin[oname]

            transform = compute_transform(
                array_size_pre=src.array_size,
                dim_pre=decoded_origin.sigs[0].n,
                array_size_post=dst.array_size,
                dim_post=dst.dimensions,
                **kwargs)

            if pstc > self.dt:
                smoothed_signals = []
                for ii in range(src.array_size):
                    filtered_signal = self.model.signal(
                        n=decoded_origin.sigs[ii].n, #-- views not ok here
                        name=decoded_origin.sigs[ii].name + '::pstc=%s' % pstc)
                    fcoef, tcoef = filter_coefs(pstc, dt=self.dt)
                    self.model.transform(tcoef,
                                         decoded_origin.sigs[ii],
                                         filtered_signal)
                    self.model.filter(fcoef, filtered_signal, filtered_signal)
                    smoothed_signals.append(filtered_signal)
                for jj in range(dst.array_size):
                    for ii in range(src.array_size):
                        if np.all(transform[ii, :, jj] == 0):
                            continue
                        self.model.filter(
                            transform[ii, :, jj].T,
                            smoothed_signals[ii],
                            dst.input_signals[jj])
            else:
                smoothed_signals = decoded_origin.sigs
                for ii in range(src.array_size):
                    for jj in range(dst.array_size):
                        if np.all(transform[ii, :, jj] == 0):
                            continue
                        self.model.transform(
                            transform[ii, :, jj].T,
                            smoothed_signals[ii],
                            dst.input_signals[jj])

        elif name1 in self.inputs:
            src = self.inputs[name1]
            dst_ensemble = self.ensembles[name2]
            if (dst_ensemble.array_size,) != src.shape:
                raise ShapeMismatch(
                    (dst_ensemble.array_size,), src.shape)

            for ii in range(dst_ensemble.array_size):
                src_ii = src[ii:ii+1]
                dst_ii = dst_ensemble.input_signals[ii]

                assert func is None
                self.model.filter(kwargs.get('transform', 1.0), src_ii, dst_ii)
        else:
            raise NotImplementedError()

    def _raw_probe(self, sig, dt_sample):
        """
        Create an un-filtered probe of the named signal,
        without constructing any filters or transforms.
        """
        return Probe(self.model.probe(sig, dt_sample), self)

    def _probe_signals(self, srcs, dt_sample, pstc):
        """
        set up a probe for signals (srcs) that will record
        their value *after* decoding, transforming, filtering.

        This is appropriate for inputs, constants, non-linearities, etc.
        But if you want to probe the part of a filter that is purely the
        decoded contribution from a population, then use
        _probe_decoded_signals.
        """
        src_n = srcs[0].size
        probe_sig = self.model.signal(
            n=len(srcs) * src_n,
            name='probe(%s)' % srcs[0].name
            )

        if pstc > self.dt:
            # -- create a new smoothed-out signal
            fcoef, tcoef = filter_coefs(pstc=pstc, dt=self.dt)
            self.model.filter(fcoef, probe_sig, probe_sig)
            for ii, src in enumerate(srcs):
                self.model.filter(tcoef, src,
                        probe_sig[ii * src_n: (ii + 1) * src_n])
            return Probe(
                self.model.probe(probe_sig, dt_sample),
                self)
        else:
            for ii, src in enumerate(srcs):
                self.model.filter(1.0, src,
                        probe_sig[ii * src_n: (ii + 1) * src_n])
            return Probe(self.model.probe(probe_sig, dt_sample),
                self)

    def _probe_decoded_signals(self, srcs, dt_sample, pstc):
        """
        set up a probe for signals (srcs) that will record
        their value just from being decoded.

        This is appropriate for functions decoded from nonlinearities.
        """
        src_n = srcs[0].size
        probe_sig = self.model.signal(
            n=len(srcs) * src_n,
            name='probe(%s)' % srcs[0].name
            )

        if pstc > self.dt:
            # -- create a new smoothed-out signal
            fcoef, tcoef = filter_coefs(pstc=pstc, dt=self.dt)
            self.model.filter(fcoef, probe_sig, probe_sig)
            for ii, src in enumerate(srcs):
                self.model.transform(tcoef, src,
                        probe_sig[ii * src_n: (ii + 1) * src_n])
            return Probe(
                self.model.probe(probe_sig, dt_sample),
                self)
        else:
            for ii, src in enumerate(srcs):
                self.model.transform(1.0, src,
                        probe_sig[ii * src_n: (ii + 1) * src_n])
            return Probe(self.model.probe(probe_sig, dt_sample),
                self)

    def make_probe(self, name, dt_sample, pstc):
        if name in self.ensembles:
            ens = self.ensembles[name]
            srcs = ens.origin['X'].sigs
            return self._probe_decoded_signals(srcs, dt_sample, pstc)

        elif name in self.inputs:
            src = self.inputs[name]
            return self._probe_signals([src], dt_sample, pstc)

        else:
            raise NotImplementedError()

    def _make_simulator(self):
        sim = self.Simulator(self.model)
        self.sim = sim

    def run(self, simtime, verbose=False):
        try:
            self.sim
        except:
            self._make_simulator()
        n_steps = int(simtime // self.dt)
        self.sim.run_steps(n_steps, verbose=verbose)
