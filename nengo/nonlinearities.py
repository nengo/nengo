import copy
import logging

import numpy as np
import scipy.signal as ss
from . import decoders

logger = logging.getLogger(__name__)


class PythonFunction(object):

    def __init__(self, fn, n_in, n_out=None, label=None):
        if label is None:
            label = "<Direct%d>" % id(self)
        self.label = label
        self.n_in = n_in
        self.fn = fn
        self.n_out = n_out

        if n_out is None:
            if self.n_args == 1:
                res = fn(np.asarray(0.0))
            elif self.n_args == 2:
                res = fn(np.asarray(0.0), np.zeros(n_in))
            self.n_out = np.asarray(res).size

    def __deepcopy__(self, memo):
        try:
            return memo[id(self)]
        except KeyError:
            rval = self.__class__.__new__(self.__class__)
            memo[id(self)] = rval
            for k, v in self.__dict__.items():
                if k == 'fn':
                    rval.fn = v
                else:
                    rval.__dict__[k] = copy.deepcopy(v, memo)
            return rval

    @property
    def n_args(self):
        return 2 if self.n_in > 0 else 1


class SurrogateFunction(PythonFunction):
    """
    Wraps a PythonFunction to add a surrogate model of noise and bias in a neuron estimate.  
    """
    def __init__(self, fn, n_in, ensemble, decoders, dt, **kwargs):
        PythonFunction.__init__(self, fn, n_in, **kwargs)

        # TODO: derive more realistic filter params from ensemble params
        noise_sd = .6*ensemble.neurons.n_neurons**-.5/100**-.5
        sigma = noise_sd**2 * np.eye(decoders.shape[0]) #noise covariance matrix
        num = [0.01594, 4.855, 160]
        den = [1, 591.4731, 252661.872668]
        
        self.noise = Noise(sigma, num, den, dt)
        if ensemble.dimensions == 1:
            self.static = InterpolatorND(.05, ensemble, decoders, dt)              
        if ensemble.dimensions == 2:
            self.static = Interpolator2D(.1, ensemble, decoders, dt)
        else:
            self.static = InterpolatorND(.1, ensemble, decoders, dt)  
        
        
class Noise(object):
    """
    Generates noise with specified covariance and spectrum.
    
    Parameters
    ----------
    sigma: covariance matrix of noise for all an ensemble's outputs (#out x #out)
    num: numerator of 2nd order transfer function that defines noise spectrum (Laplace domain)
    den: denominator of 2nd order transfer function that defines noise spectrum (Laplace domain)
    dt: time step (s) at which noise is generated
    """
    def __init__(self, sigma, num, den, dt):
        self.sigma = sigma
        self.dt = dt
        self._noise_time = np.zeros(0)
        self._zi = np.zeros([len(sigma), 2]) #initial conditions for filters
        
        self.chol_sigma = np.linalg.cholesky(sigma); 
    
        fs = 1/self.dt
        [self.bz, self.az] = ss.bilinear(num, den, fs=fs)
        
        self.make_samples(0, 3)
        actual_sd = np.std(self._noise_samples, 1)
        ideal_sd = np.sqrt(np.diag(self.sigma))
        self._gain = np.mean(ideal_sd) / np.mean(actual_sd)        


    def get_noise(self, time):
        """
        Parameters
        ---------- 
        time: end of simulation time step
         
        Returns 
        -------
        vector of noise values (a random variable with spatial and
          temporal correlations)
        """ 
        index = -1
        if len(self._noise_time) > 0:
            index = round((time - self._noise_time[0]) / self.dt)
            if (index < 0) or (index > len(self._noise_time)-1):
                index = -1
         
        if index < 0: 
            self.make_samples(time, time+1.0)
            index = 0
        
        return self._gain * self._noise_samples[:, index]

    def make_samples(self, start_time, end_time):
        """
        This method sets internal variables in support of get_noise, which calls it as needed. 
         
        Parameters
        ----------
        start_time: beginning of simulation time for which to generate noise samples
        end_time: end of simulation time for which to generate noise samples
        """
         
        previous_time = [self._noise_time[-1]] if len(self._noise_time) > 0 else []
        self._noise_time = np.arange(start_time, end_time, self.dt)
 
        n_outs = self.chol_sigma.shape[0]
        n_steps = len(self._noise_time)
         
        uncorrelated = np.random.randn(n_outs, n_steps)
        correlated = self.chol_sigma.dot(uncorrelated)
         
        if len(previous_time) == 0 or abs(start_time - previous_time[0] - self.dt) > self.dt/10:       
            self._zi = np.zeros([n_outs, 2]) #looks like a different simulation so zero initial conditions
             
        self._noise_samples = np.zeros_like(correlated);
        for i in range(n_outs):
            [self._noise_samples[i], self._zi[i]] = ss.lfilter(self.bz, self.az, correlated[i], zi=self._zi[i])


class InterpolatorND:
    """
    An interpolator that scales linearly with # dimensions by ignoring off-axis 
    nonlinearities. Results are a weighted average of interpolation along each 
    axis. 
    """ 
    def __init__(self, dx, ens, decoders, dt):
        self._dx = dx
        self._minx = -2
        self._x = np.linspace(self._minx, -self._minx, -2*self._minx/dx+1)
        self._dim = ens.dimensions
        self._outdim = decoders.shape[0]
        
        self._y = np.zeros((len(self._x), ens.dimensions, decoders.shape[0]))
        for i in range(ens.dimensions):
            x = np.zeros((len(self._x), ens.dimensions))
            x[:,i] = self._x
            r = ens.activities(eval_points=x) * dt
            self._y[:,i,:] = r.dot(decoders.T) #all dimensions of output along axis i 

        self._m = np.zeros_like(self._y)
        self._m[0:len(self._x)-1,:,:] = np.diff(self._y, axis=0) / dx #precompute slopes
        self._m[len(self._x)-1,:,:] = self._m[len(self._x)-2,:,:] #simplify later indexing  
        
    def __call__(self, x):
        if self._dim == 1: #this is treated separately for speed (about 4x faster for scalars)
            x_ind = int((x-self._minx) / self._dx)
            x_ind = x_ind if x_ind > 0 else 0
            x_ind = x_ind if x_ind < len(self._x)-1 else len(self._x)-1
            offset = x - self._x[x_ind]
            return self._y[x_ind,0,:] + offset * self._m[x_ind,0,:]
        else:
            x2 = x**2
            sx2 = sum(x2)
            xrad = sx2**0.5 * np.sign(x) 
            x_ind = np.floor((xrad-self._minx) / self._dx).astype(int)
            x_ind = np.maximum(np.minimum(x_ind, len(self._x)-1), 0)
            y_on_axes = np.zeros([self._dim, self._outdim])
            offset = xrad - self._x[x_ind]
            for i in range(self._dim):
                y_on_axes[i] = self._y[x_ind[i],i,:] + offset[i] * self._m[x_ind[i],i,:]
                 
            return x2.dot(y_on_axes) / sx2 if sx2 > 0 else np.zeros(self._outdim)


def smooth(signal, w_len):
    """
    Applies a square smoothing convolution with minimized edge effects. 
    
    Parameters
    ----------
    signal: something to smooth
    w_len: window size (odd integer)
    """
    assert w_len%2 == 1, 'Window length should be odd'
    reflected = np.concatenate((signal[w_len-1:0:-1], signal, signal[-1:-w_len:-1]), axis=1)
    return np.convolve(reflected, np.ones(w_len)/w_len, mode='same')[w_len-1:-w_len+1]

class Interpolator2D: 
    """
    Does 2D interpolation. 
    """
    def __init__(self, dx, ens, decoders, dt):
        assert ens.dimensions == 2
        self._dim = 2
        self._outdim = decoders.shape[0]
        self._dx = dx
        self._minx = -2
        self._x = np.linspace(self._minx, -self._minx, -2*self._minx/dx+1)
        nx = len(self._x)
        
        xgrid = np.tile(self._x[:,None], [1,nx])
        X = np.concatenate((np.reshape(xgrid, [1, nx**2]), np.reshape(xgrid.T, [1, nx**2])))
        r = ens.activities(eval_points=X.T) * dt
        Y = r.dot(decoders.T)  
        self._y = np.reshape(Y, [nx, nx, self._outdim]) 
        
        self._grad0 = np.subtract(self._y[1:nx,:,:], self._y[0:nx-1,:,:]) / dx
        self._grad0 = np.concatenate((self._grad0, self._grad0[nx-2:nx-1,:,:]), axis=0) #simplify later indexing
        self._grad1 = np.subtract(self._y[:,1:nx,:], self._y[:,0:nx-1,:]) / dx
        self._grad1 = np.concatenate((self._grad1, self._grad1[:,nx-2:nx-1,:]), axis=1)
        
        smoothing_window_length = 7
        for i in range(self._outdim): #smooth edges for better extrapolation 
            self._grad0[0,:,i] = smooth(self._grad0[0,:,i], smoothing_window_length)
            self._grad0[-1,:,i] = smooth(self._grad0[-1,:,i], smoothing_window_length)
            self._grad1[:,0,i] = smooth(self._grad1[:,0,i], smoothing_window_length)
            self._grad1[:,-1,i] = smooth(self._grad1[:,-1,i], smoothing_window_length)
        
    def get_index(self, x):
        x_ind = int((x-self._minx) / self._dx)
        x_ind = x_ind if x_ind > 0 else 0
        x_ind = x_ind if x_ind < len(self._x)-1 else len(self._x)-1
        return x_ind

    def __call__(self, x):
        xi0 = self.get_index(x[0])
        xi1 = self.get_index(x[1])
        offset0 = x[0]-self._x[xi0]
        offset1 = x[1]-self._x[xi1]
        return self._y[xi0, xi1,:] + offset0*self._grad0[xi0, xi1,:] + offset1*self._grad1[xi0, xi1,:] 
    
        

class Neurons(object):

    def __init__(self, n_neurons, bias=None, gain=None, label=None):
        self.n_neurons = n_neurons
        self.bias = bias
        self.gain = gain
        if label is None:
            label = "<%s%d>" % (self.__class__.__name__, id(self))
        self.label = label

    def __str__(self):
        r = self.__class__.__name__ + "("
        r += self.label if hasattr(self, 'label') else "id " + str(id(self))
        r += ", %dN)" if hasattr(self, 'n_neurons') else ")"
        return r

    def __repr__(self):
        return str(self)

    def default_encoders(self, dimensions, rng):
        raise NotImplementedError("Neurons must provide default_encoders")

    def rates(self, J_without_bias):
        raise NotImplementedError("Neurons must provide rates")

    def set_gain_bias(self, max_rates, intercepts):
        raise NotImplementedError("Neurons must provide set_gain_bias")


class Direct(Neurons):

    def __init__(self, n_neurons=None, label=None):
        # n_neurons is ignored, but accepted to maintain compatibility
        # with other neuron types
        Neurons.__init__(self, 0, label=label)

    def default_encoders(self, dimensions, rng):
        return np.eye(dimensions)

    def rates(self, J_without_bias):
        return J_without_bias

    def set_gain_bias(self, max_rates, intercepts):
        pass


# TODO: class BasisFunctions or Population or Express;
#       uses non-neural basis functions to emulate neuron saturation,
#       but still simulate very fast


class _LIFBase(Neurons):

    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, label=None):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        Neurons.__init__(self, n_neurons, label=label)

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def default_encoders(self, dimensions, rng):
        return decoders.sample_hypersphere(
            dimensions, self.n_neurons, rng, surface=True)

    def rates(self, J_without_bias):
        """LIF firing rates in Hz

        Parameters
        ---------
        J_without_bias: ndarray of any shape
            membrane currents, without bias voltage
        """
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            J = J_without_bias + self.bias
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A

    def set_gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        Parameters
        ---------
        max_rates : list of floats
            Maximum firing rates of neurons.
        intercepts : list of floats
            X-intercepts of neurons.

        """
        logging.debug("Setting gain and bias on %s", self.label)
        max_rates = np.asarray(max_rates)
        intercepts = np.asarray(intercepts)
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts


class LIFRate(_LIFBase):

    def math(self, dt, J):
        """Compute rates for input current (incl. bias)"""
        old = np.seterr(divide='ignore')
        try:
            j = np.maximum(J - 1, 0.)
            r = dt / (self.tau_ref + self.tau_rc * np.log(1 + 1. / j))
        finally:
            np.seterr(**old)
        return r


class LIF(_LIFBase):

    def __init__(self, n_neurons, upsample=1, **kwargs):
        _LIFBase.__init__(self, n_neurons, **kwargs)
        self.upsample = upsample

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # N.B. J here *includes* bias

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        old = np.seterr(all='ignore')
        try:
            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked[:] = (v > 1) * 1.0

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = (spiked * (spiketime + self.tau_ref)
                                   + (1 - spiked) * (refractory_time - dt))
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time
        
class LIFSurrogate(_LIFBase):
    """
    A surrogate model of LIF neurons. It contains neurons, but they are not meant to be simulated, 
    rather their output is meant to be approximated efficiently. 
    """
    pass

    

