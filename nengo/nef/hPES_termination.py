import collections

import numpy as np
import theano
import theano.tensor as TT

from . import neuron
from .learned_termination import LearnedTermination

class hPESTermination(LearnedTermination):
    """
    # learning_rate = 5e-7      # from nengo
    # theta_tau = 20.           # from nengo
    # scaling_factor = 20e3     # from nengo
    # supervision_ratio = 0.5   # from nengo
    """

    theta_tau = 0.02
    scaling_factor = 10.
    supervision_ratio = 1.0

    def __init__(self, *args, **kwargs):
        """
        """
        super(hPESTermination, self).__init__(*args, **kwargs)

        # get the theano instantaneous spike raster
        # of the pre- and post-synaptic neurons
        self.pre_spikes = self.pre.neurons.output
        self.post_spikes = self.post.neurons.output
        # get the decoded error signal
        self.error_value = self.error.decoded_output

        # get gains (alphas) for post neurons
        self.encoders = self.post.encoders.astype('float32')
        self.gains = np.sqrt(
            (self.post.encoders ** 2).sum(axis=-1)).astype('float32')

        self.initial_theta = np.float32(np.random.uniform(low=5e-5, high=15e-5,
            size=(self.post.array_size, self.post.neurons_num)))
        # Assumption: high gain -> high theta
        self.initial_theta *= self.gains
        self.theta = theano.shared(self.initial_theta, name='hPES.theta')

        self.pre_filtered = theano.shared(
            self.pre_spikes.get_value(), name='hPES.pre_filtered')
        self.post_filtered = theano.shared(
            self.post_spikes.get_value(), name='hPES.post_filtered')

    def reset(self):
        """
        """
        super(hPESTermination, self).reset()
        self.theta.set_value(self.initial_theta)

    def learn(self):
        """
        """
        # get the error as represented by the post neurons
        encoded_error = TT.sum(self.encoders * TT.reshape( self.error_value, 
            (self.post.array_size, 1, self.post.dimensions)) , axis=-1)

        supervised_rate = self.learning_rate
        #TODO: more efficient rewrite with theano batch command? 
        delta_supervised = [
            supervised_rate * 
            self.pre_filtered[self.pre_index(i)][None,:] *
            encoded_error[i % self.post.array_size]
            for i in range(self.post.array_size * self.pre.array_size) ]

        unsupervised_rate = TT.cast(
            self.learning_rate * self.scaling_factor, dtype='float32')
        #TODO: more efficient rewrite with theano batch command? 
        delta_unsupervised = [
            unsupervised_rate * self.pre_filtered[self.pre_index(i)][None,:] *
            ( 
                self.post_filtered[i % self.post.array_size] * 
                ( 
                    self.post_filtered[i % self.post.array_size] - 
                    self.theta[i % self.post.array_size] 
                ) * 
                self.gains[i % self.post.array_size] 
            ) for i in range(self.post.array_size * self.pre.array_size) ]

        new_wm = (self.weight_matrix
                + TT.cast(self.supervision_ratio, 'float32') * delta_supervised
                + TT.cast(1. - self.supervision_ratio, 'float32')
                * delta_unsupervised)

        return new_wm

    def pre_index(self, i): 
        """This method calculates the index of the pre-synaptic ensemble
        that should be accessed given a current index value i

        int(np.ceil((i + 1) / float(self.post.array_size)) - 1)
        generates 0 post.array_size times, then 1 post.array_size times, 
        then 2 post.array_size times, etc so with 
        pre.array_size = post.array_size = 2 we're connecting it up in order 
        [pre[0]-post[0], pre[0]-post[1], pre[1]-post[0], pre[1]-post[1]]

        :param int i: the current index value, 
            value from 0 to post.array_size * pre.array_size
        :returns: the desired pre-synaptic ensemble index
        """
        return int(np.ceil((i + 1) / float(self.post.array_size)) - 1)
        
    def update(self, dt):
        """
        """
        # update filtered inputs
        alpha = TT.cast(dt / self.pstc, dtype='float32')
        new_pre = self.pre_filtered + alpha * (
            self.pre_spikes - self.pre_filtered)
        new_post = self.post_filtered + alpha * (
            self.post_spikes - self.post_filtered)

        # update theta
        alpha = TT.cast(dt / self.theta_tau, dtype='float32')
        new_theta = self.theta + alpha * (new_post - self.theta)

        return collections.OrderedDict({
                self.weight_matrix: self.learn(),
                self.pre_filtered: new_pre, 
                self.post_filtered: new_post,
                self.theta: new_theta,
                })
