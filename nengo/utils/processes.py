from __future__ import absolute_import
import numpy as np

from nengo.utils.distributions import Distribution, Gaussian


class StochasticProcess(object):
    def __init__(self, dimensions=1):
        self.dimensions = dimensions

    def sample(self, dt, timesteps=None, rng=np.random):
        raise NotImplementedError(
            "A StochasticProcess should implement sample.")


class SampledProcess(StochasticProcess):
    def __init__(self, dist, dimensions=1):
        super(SampledProcess, self).__init__(dimensions)
        self.dist = dist

    def sample(self, dt, timesteps=None, rng=np.random):
        # FIXME correct for dt here?
        return self.dist.sample(self.dimensions, timesteps, rng=rng)


class MarkovProcess(StochasticProcess):
    def __init__(self, dist, dimensions=1, initial_state=None):
        super(MarkovProcess, self).__init__(dimensions)
        self.dist = dist
        if initial_state is None:
            self.state = np.zeros(dimensions)
        else:
            self.state = np.array(initial_state)
            if self.state.shape != (dimensions,):
                raise ValueError("initial_state has to match dimensions.")

    def sample(self, dt, timesteps=None, rng=np.random):
        samples = self.state[:, np.newaxis] + np.cumsum(
            self.dist.sample(
                self.dimensions, timesteps, rng=rng) * np.sqrt(dt),
            axis=0 if timesteps is None else 1)
        self.state[:] = samples[:, -1]
        return samples[:, 0] if timesteps is None else samples


class WienerProcess(MarkovProcess):
    def __init__(self, dimensions=1, initial_state=None):
        super(WienerProcess, self).__init__(
            Gaussian(0, 1.), dimensions, initial_state)
