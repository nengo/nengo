import collections
import numpy as np
from .neuron import Neuron

class LIFNeuron(Neuron):
    def __init__(self, neurons, dimensions, tau_rc=0.02, tau_ref=0.002):
        """Constructor for a set of LIF rate neuron.

        :param int size: number of neurons in set
        :param float dt: timestep for neuron update function
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        :param int neurons: number of neurons in this population
        :param int dimensions:
            number of dimensions in the vector space
            that these neurons represent
        :param float tau_ref: length of refractory period
        :param float tau_rc:
            RC constant; approximately how long until 2/3
            of the threshold voltage is accumulated

        """
        Neuron.__init__(self, neurons * dimensions)
        self.tau_rc = tau_rc
        self.tau_ref  = tau_ref
        self.voltage = np.zeros((neurons, dimensions), 'float32')
        self.refractory_time = np.zeros((neurons, dimensions), 'float32')
        
    def make_alpha_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        x = 1.0 / (1 - np.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        alpha = (1 - x) / (intercepts - 1.0)
        j_bias = 1 - alpha * intercepts
        return alpha, j_bias

    # TODO: have a reset() function at the ensemble and network level
    #that would actually call this
    def reset(self):
        """Resets the state of the neuron."""
        Neuron.reset(self)

        self.voltage.set_value(np.zeros(self.size).astype('float32'))
        self.refractory_time.set_value(np.zeros(self.size).astype('float32'))

    def update(self, J, dt):
        """Theano update rule that implementing LIF rate neuron type
        Returns dictionary with voltage levels, refractory periods,
        and instantaneous spike raster of neurons.

        :param float array J:
            the input current for the current time step
        :param float dt: the timestep of the update
        """

        # Euler's method
        dV = dt / self.tau_rc * (J - self.voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(self.voltage + dV, 0)  
        
        # handle refractory period        
        post_ref = 1.0 - (self.refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)
        
        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked = np.switch(v > 1, 1.0, 0.0)
        
        # adjust refractory time (neurons that spike get
        # a new refractory time set, all others get it reduced by dt)

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV 
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        new_refractory_time = np.switch(
            spiked, spiketime + self.tau_ref, self.refractory_time - dt)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        self.voltage[:] = v * (1 - spiked)
        self.refractory_time[:] = new_refractory_time
        self.output[:] = spiked

