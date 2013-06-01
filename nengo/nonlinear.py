import numpy as np

#
# Definitions of standard kinds of Non-Linearity

# TODO: Consider moving these into simulator_objects.py
# because they are the basic objects that are destinations for
# e.g. encoders and sources for decoders. They are tightly
# part of that set of objects.
#

class Direct(object):
    """
    """
    def __init__(self, n_in, n_out, fn):
        """
        fn: 
        """
        self.n_in = n_in
        self.n_out = n_out
        self.fn = fn

    def fn(self, J):
        return J


class LIF(object):
    def __init__(self, n_neurons, tau_rc=0.02, tau_ref=0.002, upsample=1):
        self.n_neurons = n_neurons
        self.upsample = upsample
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = np.random.rand(n_neurons)
        self.bias = np.random.rand(n_neurons)

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons

    def set_gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons

        """
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - self.gain * intercepts

    def step_math0(self, dt, J, voltage, refractory_time, spiked):
        if self.upsample != 1:
            raise NotImplementedError()

        # Euler's method
        dV = dt / self.tau_rc * (J + self.bias - voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (refractory_time - dt) / dt

        # set any post_ref elements < 0 = 0, and > 1 = 1
        v *= np.clip(post_ref, 0, 1)

        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked[:] = (v > 1) * 1.0

        old = np.seterr(all='ignore')
        try:

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            new_refractory_time = spiked * (spiketime + self.tau_ref) \
                                  + (1 - spiked) * (refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        voltage[:] = v * (1 - spiked)
        refractory_time[:] = new_refractory_time

    def rates(self, J):
        """Return LIF firing rates for current J in Hz

        Parameters
        ---------
        J: ndarray of any shape
            membrane voltages
        tau_rc: broadcastable like J
            XXX
        tau_ref: broadcastable like J
            XXX
        """
        old = np.seterr(all='ignore')
        J = J + self.bias
        try:
            A = self.tau_ref - self.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A


class LIFRate(LIF):

    @property
    def n_in(self):
        return self.n_neurons

    @property
    def n_out(self):
        return self.n_neurons



