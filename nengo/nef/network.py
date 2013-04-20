import random
import collections
import quantities

import numpy as np

from . import ensemble
from . import simplenode
from . import probe
from . import origin
from . import input
from . import subnetwork
from . import connections
from . import learning_rule

class Network(object):
    def __init__(self, name, seed=None, fixed_seed=None):
        """Wraps an NEF network with a set of helper functions
        for simplifying the creation of NEF models.

        :param string name:
            create and wrap a new Network with the given name.
        :param int seed:
            random number seed to use for creating ensembles.
            This one seed is used only to start the
            random generation process, so each neural group
            created will be different.

        """
        self.name = name
        self.seed = seed
        self.fixed_seed = fixed_seed
        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)
        # dictionaries for the objects in the network
        self.Connections = []
        self.Ensembles = []
        self.Networks = []
        self.Nodes = []
        self.Probes = []
          
    def add(self, object):
        """Add an object to the network to the appropriate list.
        
        :param object: the object to add to this network
        :param type: Network, Node, Ensemble, Connection

        """
        if isinstance(object, connection.Connection): self.Connections.append(object)
        elif isinstance(object, ensemble.Ensemble): self.Ensembles.append(object)
        elif isinstance(object, network.Network): self.Network.append(object)
        elif isinstance(object, node.Node): self.Node.append(object)
        elif isinstance(object, probe.Probe): self.Probes.append(object) 
        else: raise Exception('Object type not recognized')

    def compute_transform(self, dim_pre, dim_post, array_size, weight=1,
                          index_pre=None, index_post=None, transform=None):
        """Helper function used by :func:`nef.Network.connect()` to create
        the `dim_post` by `dim_pre` transform matrix.

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
        
    def connect(self, pre, post, transform=None, function=None,
                filter=None, learning_rule=None):
        """Connect two nodes in the network.

        *pre* and *post* can be strings giving the names of the nodes,
        or they can be the nodes themselves (Inputs and Ensembles are
        supported). They can also be actual Origins or Terminations,
        or any combination of the above. 

        If transform is None, it defaults to the identity matrix.
        
        transform must be of size post.dimensions * pre.dimensions.

        If *func* is not None, a new Origin will be created on the
        pre-synaptic ensemble that will compute the provided function.
        The name of this origin will be taken from the name of
        the function.

        :param string pre: Name of the node to connect from.
        :param string post: Name of the node to connect to.
        :param transform:
            The linear transfom matrix to apply across the connection.
            If *transform* is T and *pre* represents ``x``,
            then the connection will cause *post* to represent ``Tx``.
            Should be an N by M array, where N is the dimensionality
            of *post* and M is the dimensionality of *pre*.
        :type transform: array of floats
        :param Filter filter: the filter
        :param LearningRule learning_rule: a learning rule to use to update weights
        :param function func:
            Function to be computed by this connection.
            If None, computes ``f(x)=x``.
            The function takes a single parameter ``x``, which is
            the current value of the *pre* ensemble, and must return
            either a float or an array of floats.

        """

        # get pre Node object from node dictionary
        pre = self.get_object(pre)

        # get post Node object from node dictionary
        post = self.get_object(post)

        # get the origin from the pre Node
        pre_origin = self.get_origin(pre, function)

        # get decoded_output from specified origin
        pre_output = pre_origin.decoded_output
        dim_pre = pre_origin.dimensions 
        
        # if decoded-decoded connection (case 1)
        # compute transform if not given, if given make sure shape is correct
        if transform is None:
            transform = self.compute_transform(
                dim_pre=dim_pre,
                dim_post=post.dimensions,
                array_size=post.array_size)

        # pass in the pre population decoded output function
        # to the post population
        c = Connection(pre=pre, post=post, transform=transform, filter=filter,
                       function=function, learning_rule=learning_rule)
        self.add(c)
        return c

    def connect_neurons(self, pre, post, weights=None, filter=None, learning_rule=None):
        """Connect two nodes in the network, directly specifying the weight matrix

        *pre* and *post* can be strings giving the names of the nodes,
        or they can be the nodes themselves (Inputs and Ensembles are
        supported).

        :param string pre: Name of the node to connect from.
        :param string post: Name of the node to connect to.
        :param weights: The connection weight matrix between the populations.
        :type weights: An (pre.dimensions x post.dimensions) array of floats
        :param Filter filter: the filter
        :param LearningRule learning_rule: a learning rule to use to update weights
        """

        # dereference pre- and post- strings if necessary
        pre = self.get_object(pre)
        post = self.get_object(post)

        # create connection
        c = Connection(pre=pre, post=post, weights=weights,
                       filter=filter, learning_rule=learning_rule)
        self.add(c)
        return c

    def get_object(self, name):
        """This is a method for parsing input to return the proper object.

        The only thing we need to check for here is a ':',
        indicating an origin.

        :param string name: the name of the desired object
        
        """
        assert isinstance(name, str)

        # separate into node and origin, if specified
        split = name.split(':')

        if len(split) == 1:
            # no origin specified
            return self.nodes[name]

        elif len(split) == 2:
            # origin specified
            node = self.nodes[split[0]]
            return node.origin[split[1]]
       
    def get_origin(self, name, func=None):
        """This method takes in a string and returns the decoded_output function 
        of this object. If no origin is specified in name then 'X' is used.

        :param string name: the name of the object(and optionally :origin) from
                            which to take decoded_output from
        :returns: specified origin
        """
        obj = self.get_object(name) # get the object referred to by name

        if not isinstance(obj, origin.Origin):
            # if obj is not an origin, find the origin
            # the projection originates from

            # take default identity decoded output from obj population
            origin_name = 'X'

            if func is not None: 
                # if this connection should compute a function

                # set name as the function being calculated
                origin_name = func.__name__

                #TODO: better analysis to see if we need to build a new origin
                # (rather than just relying on the name)
                if origin_name not in obj.origin:
                    # if an origin for this function hasn't already been created
                    # create origin with to perform desired func
                    obj.add_origin(origin_name, func, dt=self.dt)

            obj = obj.origin[origin_name]

        else:
            # if obj is an origin, make sure a function wasn't given
            # can't specify a function for an already created origin
            assert func == None

        return obj

    def make_ensemble(self, name, neurons, dimensions, max_rate_uniform=(50,100),
                      intercept_uniform=(-1,1), radius=1, encoders=None): 
        """Create and return an ensemble of neurons.

        :param string name: name of the ensemble (must be unique)
        :param int neurons: number of neurons in the ensemble
        :param int dimensions: number of dimensions the ensemble represents
        :param tuple max_rate_uniform: distribution of max firing rates of the neurons in the ensemble
        :param tuple intercept_uniform: distribution of neuron intercepts
        :param float radius: radius
        :param list encoders: the encoders
        :returns: the newly created ensemble

        """
        e = ensemble.Ensemble(name, neurons=neurons, dimensions=dimensions,
            max_rate_uniform=max_rate_uniform, intercept_uniform=intercept_uniform,
            radius=radius, encoders=encoders)

        # store created ensemble in node dictionary
        self.add(e)
        return e

    def make_node(self, name, output): 
        """Create a Node and add it to the network."""
        n = node.Node(name, output=output)
        self.add(n)
        return n
        
    def make_network(self, name):
        """Create a subnetwork.  This has no functional purpose other than
        to help organize the model.  Components within a subnetwork can
        be accessed through a dotted name convention, so an element B inside
        a subnetwork A can be referred to as A.B.       
        
        :param name: the name of the subnetwork to create        
        """
        return self.add(network.Network(name, self))
            
    def make_probe(self, target, sample_every=0.01, static=False):
        """Add a probe to measure the given target.
        
        :param target: a variable to record
        :param sample_every: the sampling frequency of the probe
        :param static: True if this variable should only be sampled once.
        :returns: The Probe object
        
        """
        i = 0
        name = None
        while name is None or self.nodes.has_key(name):
            i += 1
            name = ("Probe%d" % i)

        # get the signal to record
        if data_type == 'decoded':
            target = self.get_origin(target).decoded_output

        elif data_type == 'spikes':
            target = self.get_object(target)
            # check to make sure target is an ensemble
            assert isinstance(target, ensemble.Ensemble)
            target = target.neurons.output

        p = probe.Probe(name=name, target=target, sample_every=sample_every, static=static)
        self.add(p)
        return p
            
