import numpy as np

from .filter import Filter

class Connection(object):
    
    def __init__(self, pre, post, 
                 transform=None, function=None, weights=None,
                 filter=Filter(), learning_rule=None):
        vector_space = transform is not None or function is not None
        neuron_space = weights is not None
        if vector_space and neuron_space:
            raise ValueError(
                "Cannot provide both vector_space arguments (transform and function)" + 
                " and neuron space arguments (weights).")
            
        if vector_space:
            return VectorConnection()
        
        if neuron_space:
            return NeuronConnection()
        

class VectorConnection(Connection):
    
    def __init__(self, pre, post, 
                 transform=None, function=None, 
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
        :param filter: a Filter object describing the post-synaptic filtering
            properties of the connection
        :param learning_rule: a LearningRule object describing the learning
            rule to use with the population
        """
        ### basic parameters, set by network.connect(...)
        self.pre = pre
        self.post = post
        self.transform = transform
        self.function = function
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

    def compute_transform(self, dim_pre, dim_post, array_size, weight=1,
                          index_pre=None, index_post=None, transform=None):
        """Helper function used by :func:`Network.connect()` to create
        the `dim_pre` by `dim_post` transform matrix.

        Values are either 0 or *weight*. *index_pre* and *index_post*
        are used to determine which values are non-zero, and indicate
        which dimensions of the pre-synaptic ensemble should be routed
        to which dimensions of the post-synaptic ensemble.

        :param int dim_pre: first dimension of transform matrix
        :param int dim_post: second dimension of transform matrix
        :param int array_size: size of the network array
        :param float weight: the non-zero value to put into the matrix
        :param index_pre: the indexes of the pre-synaptic dimensions to use
        :type index_pre: list of integers or a single integer
        :param index_post:
            the indexes of the post-synaptic dimensions to use
        :type index_post: list of integers or a single integer
        :returns:
            a two-dimensional transform matrix performing
            the requested routing

        """

        if transform is None:
            # create a matrix of zeros
            transform = [[0] * dim_pre for i in range(dim_post * array_size)]

            # default index_pre/post lists set up *weight* value
            # on diagonal of transform
            
            # if dim_post * array_size != dim_pre,
            # then values wrap around when edge hit
            if index_pre is None:
                index_pre = range(dim_pre) 
            elif isinstance(index_pre, int):
                index_pre = [index_pre] 
            if index_post is None:
                index_post = range(dim_post * array_size) 
            elif isinstance(index_post, int):
                index_post = [index_post]

            for i in range(max(len(index_pre), len(index_post))):
                pre = index_pre[i % len(index_pre)]
                post = index_post[i % len(index_post)]
                transform[post][pre] = weight

        transform = np.array(transform)

        # reformulate to account for post.array_size
        if transform.shape == (dim_post * array_size, dim_pre):

            array_transform = [[[0] * dim_pre for i in range(dim_post)]
                               for j in range(array_size)]

            for i in range(array_size):
                for j in range(dim_post):
                    array_transform[i][j] = transform[i * dim_post + j]

            transform = array_transform

        return transform
    
    def get_transformed_pre(self, state, dt):
        """Returns the transformed, filtered value output from pre."""
        
        pre_in = state[self.pre]
        pre_in = self.transform*pre_in
        pre_in = self.filter.update(dt, source=pre_in)
        return input
    
#    def weights

class NeuronConnection:
    def __init__(self, pre, post, weights=None, 
                 filter=Filter(), learning_rule=None):
        """
        Create a new connection between two objects. This connection should
        be added to a common parent of the objects.
        
        :param pre: pre-population object
        :param post: post-population object
        :param weights: neuron-space connection weight matrix describing the mapping 
            between the pre-population and post-population neurons
        :param filter: a Filter object describing the post-synaptic filtering
            properties of the connection
        :param learning_rule: a LearningRule object describing the learning
            rule to use with the population
        """
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
    
    def get_transformed_pre(self, state, dt):
        """Returns the transformed, filtered value output from pre."""
        
        pre_in = state[self.pre]
        pre_in = self.weights*pre_in
        pre_in = self.filter.update(dt, source=pre_in)
        return input