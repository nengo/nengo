
class StochasticProcess(object):
    pass


class GaussianProcess(StochasticProcess):

    def __init__(self, sigma, synapse=None):
        self.sigma = sigma
        self.synapse = synapse
