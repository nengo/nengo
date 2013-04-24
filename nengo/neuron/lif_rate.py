import collections

import numpy as np

from .neuron import Neuron

# an example of implementing a rate-mode neuron

class LIFRateNeuron(Neuron):
    def __init__(self, size, tau_rc=0.02, tau_ref=0.002):
        """Constructor for a set of LIF rate neuron

        :param int size: number of neurons in set
        :param float t_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        Neuron.__init__(self, size)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    def _build(self, state, dt):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.
        
        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        
        """
        x = 1.0 / (1 - np.exp(
                (self.tau_ref - (1.0 / self.max_rates)) / self.tau_rc))
        self.alpha = (1 - x) / (self.intercepts - 1.0)
        self.j_bias = 1 - self.alpha * self.intercepts

        state[self.output] = np.zeros((self.size, 1))

        return self.alpha, self.j_bias

    def _step(self, new_state, J, dt):
        """Update rule that implements LIF rate neuron type.
        
        Returns array with firing rates for current time step.

        """
        # set up denominator of LIF firing rate equation
        rate = self.tau_ref - self.tau_rc * np.log(
            1 - 1.0 / np.maximum(J, 0))
        
        # if input current is enough to make neuron spike,
        # calculate firing rate, else return 0
        rate = 1 / rate
        rate[J <= 1] = 0

        new_state[self.output] = rate
        return rate
