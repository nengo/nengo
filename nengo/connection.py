import numpy as np

from filter import make_filter

def make_connection(pre, post, 
                 transform=None, function=None, weights=None,
                 filter=None, learning_rule=None):
    """
    Create a new connection between two objects. This connection should
    be added to a common parent of the objects.
    
    :param pre: pre-population output object
    :param post: post-population object
    :param transform: vector-space transform matrix describing the mapping 
        between the pre-population and post-population dimensions
    :type transform: a (pre.dimensions x post.dimensions) array of floats
    :param function: the vector-space function to be computed by the 
        pre-population decoders
    :param weights: neuron-space connection weight matrix describing the mapping 
        between the pre-population and post-population neurons
    :param filter: a Filter object describing the post-synaptic filtering
        properties of the connection
    :param learning_rule: a LearningRule object describing the learning
        rule to use with the population
    """
    if (transform != None or function != None) and weights != None:
        raise ValueError(
            "Cannot provide both vector space arguments (transform and function)" + 
            " and neuron space arguments (weights).")

    if filter: 
        filter = make_filter(filter)
        
    if weights == None: 
        return VectorConnection(pre, post, transform, function, filter, learning_rule)
    else:
        return NeuronConnection(pre, post, weights, filter, learning_rule)
    
class Connection():
    pass
        
class VectorConnection(Connection):
    
    def __init__(self, pre, post, 
                 transform=None, function=None, 
                 filter=None, learning_rule=None):
        
        ### basic parameters, set by network.connect(...)
        self.pre = pre
        self.post = post
        self.function = function
        self.filter = filter
        self.learning_rule = learning_rule
        self.transform = np.array(transform)

        ### additional (advanced) parameters
        self._modulatory = False

    @property
    def modulatory(self):
        """Setting \"modulatory\" to True stops the connection from imparting
        current on the post-population."""
        return self._modulatory

    @modulatory.setter
    def modulatory(self, value):
        self._modulatory = value

    
    def get_post_input(self, state, dt):
        """Returns the transformed, filtered value output from pre."""
        pre_in = state[self.pre]
        if self.transform:
            pre_in = np.dot(self.transform, pre_in)
        '''if self.filter: 
            pre_in = self.filter.filter(dt, source=pre_in)'''
        return pre_in
    
    def learn(self, dt):
        if self.learning_rule: 
            self.learning_rule.update_weights(dt)

class NeuronConnection:
    def __init__(self, pre, post, weights, 
                 filter=None, learning_rule=None):
        ### basic parameters, set by network.connect(...)
        self.pre = pre
        self.post = post
        self.weights = weights
        self.filter = filter
        self.learning_rule = learning_rule

        ### additional (advanced) parameters
        self._modulatory = False

    @property
    def modulatory(self):
        """Setting \"modulatory\" to True stops the connection from imparting
        current on the post-population."""
        return self._modulatory

    @modulatory.setter
    def modulatory(self, value):
        self._modulatory = value
    
    def get_post_input(self, state, dt):
        """Returns the transformed, filtered value output from pre."""
        
        pre_in = state[self.pre]
        pre_in = self.weights*pre_in
        if self.filter: 
            pre_in = self.filter.update(dt, source=pre_in)
        return pre_in
    
    def learn(self, dt):
        self.learning_rule.update_weights(dt)
