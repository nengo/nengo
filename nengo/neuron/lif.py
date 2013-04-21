import collections
import numpy as np
from .neuron import Neuron

class LIFNeuron(Neuron):
    def __init__(self, tau_rc=0.02, tau_ref=0.002,
                 size=None, max_rate=None, intercept=None, seed=None):
        """Constructor for a set of LIF rate neuron.

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
        self.tau_rc = tau_rc
        self.tau_ref  = tau_ref

        self.size = size
        self.max_rate = max_rate
        self.intercept = intercept
        self.seed = seed

        self.J = None # this is set by the Ensemble
        self.voltage = Output()
        self.refractory_time = Output()
        self.output = Output()
        self.alpha = Output()
        self.j_bias = Output()

    @property
    def default_output(self):
        return self.spikes

    def make_alpha_bias(self, ):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        return alpha, j_bias

    def _build(self, state, dt):
        self.rng = np.random.RandomState(seed=self.seed)
        self.max_rate = max_rate
        max_rates = self.rng.uniform(size=self.size,
            low=max_rate[0], high=max_rate[1])  
        threshold = self.rng.uniform(self.size,
            low=intercept.low, high=intercept.high)
        x = 1.0 / (1 - np.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        state[self.alpha] = (1 - x) / (intercepts - 1.0)
        state[self.j_bias] = 1 - alpha * intercepts
        return self._reset(self, state)

    def _reset(self, state):
        state[self.voltage] = np.zeros(self.size)
        state[self.refractory_time] = np.zeros(self.size)
        state[self.output] = np.zeros(self.size)

    def _step(self, old_state, new_state, dt):
        """Theano update rule that implementing LIF rate neuron type
        Returns dictionary with voltage levels, refractory periods,
        and instantaneous spike raster of neurons.

        :param float array J:
            the input current for the current time step
        :param float dt: the timestep of the update
        """

        J = old_state[self.J]
        voltage = old_state[self.voltage]
        refractory_time = old_state[self.refractory_time]

        # Euler's method
        dV = dt / self.tau_rc * (J - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)  
        
        # handle refractory period        
        post_ref = 1.0 - (refractory_time - dt) / dt

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
            spiked, spiketime + self.tau_ref, refractory_time - dt)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        new_state[self.voltage] = v * (1 - spiked)
        new_state[self.refractory_time] = new_refractory_time
        new_state[self.output] = spiked

