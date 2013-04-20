

from .filter import Filter

class Connection(object):
    """A connection between two objects (Ensembles, Nodes, Networks)

    This class describes a connection between two objects. It contains
    the source (pre-population) and destination (post-population) of
    the connection. It also contains information about the computation of
    the connection, including functions or dimension transforms. Alternatively,
    it can represent a direct neuron-to-neuron connection by passing in a
    weight matrix. Finally, the connection can perform learning if given a
    learning rule.
    """
    
    def __init__(self, pre, post, 
                 transform=None, function=None, weights=None,
                 filter=Filter(), learning_rule=None):
        """
        Create a new connection between two objects. This connection should
        be added to a common parent of the objects.
        
        :param pre: pre-population object
        :param post: post-population object
        :param transform: vector-space transform matrix describing the mapping 
            between the pre-population and post-population dimensions
        :type transform: a (pre.dimensions x post.dimensions) array of floats
        :param function: the vector-space function to be computed by the 
            pre-population decoders
        :param weights: the connection-weight matrix for connecting pre-neurons
            to post-neurons directly. Cannot be used with transform or function.
        :type weights: a (pre.neurons x post.neurons) array of floats
        :param filter: a Filter object describing the post-synaptic filtering
            properties of the connection
        :param learning_rule: a LearningRule object describing the learning
            rule to use with the population
        """
        vector_space = transform is not None or function is not None
        neuron_space = weights is not None
        if vector_space and neuron_space:
            raise ValueError(
                "Cannot provide both vector_space arguments (transform and function)" + 
                " and neuron space arguments (weights).")

        ### basic parameters, set by network.connect(...)
        self.pre = pre
        self.post = post
        self.transform = transform
        self.function = function
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

