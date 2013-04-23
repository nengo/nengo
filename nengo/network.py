import random
import collections
#import quantities

import numpy as np

from . import ensemble
from . import probe
from . import origin
from . import input
from . import node

from ensemble import SpikingEnsemble
from output import Output
from connection import Connection
import nengo

class Network(object):
    def __init__(self, name):
        """Wraps an NEF network with a set of helper functions
        for simplifying the creation of NEF models.

        :param string name:
            create and wrap a new Network with the given name.

        """
        self.name = name

        # metadata and properties
        self._metadata = {}
        self._properties = {}
        
        # dictionaries for the objects in the network
        self.connections = []
        self.ensembles = []
        self.networks = []
        self.nodes = []
        self.probes = []

    @property
    def all_nodes(self):
        # XXX make this recursive
        return self.nodes

    @property
    def all_ensembles(self):
        # XXX make this recursive
        return self.ensembles

    @property
    def all_probes(self):
        # XXX make this recursive
        return self.probes
    
    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        
    @property
    def properties(self):
        # Compute statistics here (like neuron count, etc)
        return self._properties
    
    def add(self, object):
        """Add an object to the network to the appropriate list.
        
        :param object: the object to add to this network
        :param type: Network, Node, Ensemble, Connection

        """
        if   ensemble.is_ensemble(object): self.ensembles.append(object)
        elif isinstance(object, Network): self.network.append(object)
        elif node.is_node(object): self.nodes.append(object)
        elif probe.is_probe(object): self.probes.append(object) 
        else:
            raise TypeError('Object type not recognized', object)

    def connect(self, pre, post, transform=None, filter=None, 
                func=None, learning_rule=None):
        """Connect two objects in the network.

        *pre* and *post* can be strings giving the names of the objects,
        or they can be the objects themselves (Nodes and Ensembles are
        supported).

        If transform is None, it defaults to the identity matrix.
        
        transform must be of size post.dimensions * pre.dimensions.

        If *func* is not None, a new Origin will be created on the
        pre-synaptic ensemble that will compute the provided function.
        The name of this origin will be taken from the name of
        the function.

        :param pre: Name of the object to connect from, or object itself.
        :param post: Name of the object to connect to, or the object itself.
        :param transform:
            The linear transfom matrix to apply across the connection.
            If *transform* is T and *pre* represents ``x``,
            then the connection will cause *post* to represent ``Tx``.
            Should be an N by M array, where N is the dimensionality
            of *post* and M is the dimensionality of *pre*.
        :type transform: array of floats
        :param dict filter: dictionary describing the desired filter
        :param LearningRule learning_rule: a learning rule to use to update weights
        :param function func:
            Function to be computed by this connection.
            If None, computes ``f(x)=x``.
            The function takes a single parameter ``x``, which is
            the current value of the *pre* ensemble, and must return
            an array of floats.

        """

        # get pre object from node dictionary
        if isinstance(pre, ""):
            pre = self.get(pre)

        # get post object from node dictionary
        if isinstance(pre, ""):
            post = self.get(post)

        #use identity function if func not given
        if func == None:
            def X(val):
                return val
            func = X

        if not isinstance(pre, Output):
            #add new output to pre with given function
            o = pre.add_output(func, dimensions=func(pre.neurons.output).shape[0], name=func.__name__)
        else:
            o = pre

        
        
        # compute identity transform if no transform given
        if transform is None:
            dim_pre = o.dimensions 
            transform = nengo.gen_transform(
                dim_pre=dim_pre,
                dim_post=post.dimensions)

        #create connection
        c = Connection(pre=o, post=post, transform=transform, filter=filter,
                       function=func, learning_rule=learning_rule)
        post.add_connection(c)
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
        """Returns the ensemble, node, or network with given name."""
        search = [x for x in self.nodes+self.ensembles+self.networks if x.name == name]
        if len(search) > 0:
            print "Warning, found more than one object with same name"
        return search[0]
        
    def get(self, name, func=None):
        """This method takes in a string and returns the corresponding object.

        :param string name: the name of the object
        :returns: specified origin
        """
        if not isinstance(name, str):
            return name

        # separate into node and origin, if specified
        split = name.split(':')

        target = self.get_object(split[0])

        if len(split) == 2:
            # origin specified
            target = target.outputs[split[1]]
#        if not isinstance(obj, origin.Origin):
#            # if obj is not an origin, find the origin
#            # the projection originates from
#
#            # take default identity decoded output from obj population
#            origin_name = 'X'
#
#            if func is not None: 
#                # if this connection should compute a function
#
#                # set name as the function being calculated
#                origin_name = func.__name__
#
#            #TODO: better analysis to see if we need to build a new origin
#            # (rather than just relying on the name)
#            if origin_name not in obj.origin:
#                # if an origin for this function hasn't already been created
#                # create origin with to perform desired func
#                obj.add_origin(origin_name, func, dt=self.dt)
#
#            obj = obj.origin[origin_name]
#
#        else:
#            # if obj is an origin, make sure a function wasn't given
#            # can't specify a function for an already created origin
#            assert func == None

        return obj

    def make_alias(self, name, target):
        """ Set up an alias for referencing the target object.
        """
        self.aliases[name] = target

    def make_ensemble(self, name, neurons, dimensions, max_rate=(50,100),
                      intercept=(-1,1), radius=1, encoders=None): 
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
        # TODO use name
        e = SpikingEnsemble(neurons=neurons, dimensions=dimensions,
                              max_rate=Uniform(*max_rate),
                              intercept=Uniform(*intercept),
                              radius=radius, encoders=encoders)

        # store created ensemble in node dictionary
        self.add(e)
        return e

    def make_node(self, name, output): 
        """Create a Node and add it to the network.

        Output can be either a function or a np.ndarray.
        If it's a function, it should return a np.ndarray.
        It will called like this:
            output(simtime)

        Any arguments your function needs should be retrieved via a closure or
        self or something.
        
        """
        n = node.Node(name, output=output)
        self.add(n)
        return n
        
    def make_network(self, name, seed=None):
        """Create a subnetwork.  This has no functional purpose other than
        to help organize the model.  Components within a subnetwork can
        be accessed through a dotted name convention, so an element B inside
        a subnetwork A can be referred to as A.B.       
        
        :param name: the name of the subnetwork to create        
        """
        return self.add(network.Network(name, self))

    def probe(self, target, sample_every=0.01, static=False):
        """Add a probe to measure the given target.
        
        :param target: a variable to record
        :param sample_every: the sampling frequency of the probe
        :param static: True if this variable should only be sampled once.
        :returns: The Probe object
        
        """
        p = probe.ListProbe(target=target,
                        sample_every=sample_every,
                        static=static)
        self.Probes.append(p)

    def remove(self, obj):
        """Removes an object from the network.
        
        :param obj: The object to remove
        :param type: <nengo string> or Ensemble, Node, Network, Connection
        """

