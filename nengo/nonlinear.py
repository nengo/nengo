import numpy as np


class Direct(object):
    def __init__(self, n):
        self.n = n

    def step(self, input, dt, output):
        output[...] = input[...]

    def step_opencl(self, input, dt, output):
        raise NotImplementedError()


class LIFRate(LIF):
    def step(self, J, dt, output):
        output[...] = self._batch_lif_rates(J) * dt

    def _batch_lif_rates(self, J):
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
        try:
            A = self.lif.tau_ref - self.lif.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A

    def step_opencl(self, J, dt, spiked):
        raise NotImplementedError()


class LIF(object):
    def __init__(self, neurons, tau_rc=0.02, tau_ref=0.002, upsample=1):
        self.upsample = upsample
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gain = None
        self.bias = None

    def set_gain_bias(self, max_rates, intercepts):
        x = 1.0 / (1 - np.exp(
            (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        self.gain = (1 - x) / (intercepts - 1.0)
        self.bias = 1 - alpha * intercepts

    def step(self, J, dt, spiked):
        if self.lif.upsample != 1:
            raise NotImplementedError()

        # Euler's method
        dV = dt / self.lif.tau_rc * (J - self.voltage)

        # increase the voltage, ignore values below 0
        v = np.maximum(self.voltage + dV, 0)

        # handle refractory period
        post_ref = 1.0 - (self.refractory_time - dt) / dt

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
            new_refractory_time = spiked * (spiketime + self.lif.tau_ref) \
                                  + (1 - spiked) * (self.refractory_time - dt)
        finally:
            np.seterr(**old)

        # return an ordered dictionary of internal variables to update
        # (including setting a neuron that spikes to a voltage of 0)

        self.voltage[:] = v * (1 - spiked)
        self.refractory_time[:] = new_refractory_time

    def step_opencl(self, J, dt, spiked):
        raise NotImplementedError()


# Nonlinearities must register themselves here so that
# the simulator can recognize their type.
registry = [Direct, LIF, LIFRate]
