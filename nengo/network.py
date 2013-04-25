import inspect
import numpy as np

from . import ensemble
from . import probe
from . import node

from ensemble import SpikingEnsemble
from output import Output
import connection
import nengo

class Network():
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

    def connect(self, pre, post, transform=None, 
                filter=nengo.pstc(0.01), 
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

        # get post object from node dictionary
        if isinstance(post, basestring):
            postname = post.split("/")[-1]
            post = self.get(post)
        else:
            postname = post.name

        o = self.get_object_output(pre, func=func)
        
        #create connection
        c = connection.make_connection(pre=o, post=postname, transform=transform, 
                        filter=filter, function=func, 
                        learning_rule=learning_rule)

        post.add_connection(c)
        
        return c

    def connect_neurons(self, pre, post, weights=None, 
                        filter=None, learning_rule=None):
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

        pass

    def get(self, name, func=None):
        """This method takes in a string and returns the corresponding object.

        :param string name: the name of the object
        :returns: specified origin
        """
        if not isinstance(name, str):
            return name

        #recursively get from subnetworks
        if "/" in name:
            subsplit = name.split("/")
            target = self.get(subsplit[0]).get("/".join(subsplit[1:]))
        else:
            target = self.get_object(name.split(":")[0])
        
        # load specific input/output if given
        if ":" in name:
            target = target.get(name)

        return target

    def get_object_output(self, obj, func=None):
        """
        """
        # get object from node dictionary
        if isinstance(obj, basestring):
            obj = self.get(obj)

        if not isinstance(obj, Output):
            #then the obj is an ensemble or node (i.e. no specific output
            #was specified)
            
            #try to load default output
            o = obj.get(obj.name + ":output")
            
            if o is None:
                #we need to create a new output and add it to the object

                #use identity function if func not given
                if func == None:
                    def output(val):
                        return val
                    func = output
                
                #add new output to pre with given function
                o = obj.add_output(func)
        else:
            o = obj 
        return o

    def get_object(self, name):
        """Returns the ensemble, node, or network with given name."""
        search = [x for x in self.nodes+self.ensembles+self.networks if x.name == name]
        
        if len(search) > 1:
            print "Warning, found more than one object with same name"
        if len(search) == 0:
            print name + " not found in network.get_object"
            return None
        return search[0]
       
    def make_alias(self, name, target):
        """ Set up an alias for referencing the target object.
        """
        self.aliases[name] = target

    def make_ensemble(self, name, num_neurons, dimensions, max_rate=(50,100),
                      intercept=(-1,1), radius=1, encoders=None): 
        """Create and return an ensemble of neurons.

        :param string name: name of the ensemble (must be unique)
        :param int num_neurons: number of neurons in the ensemble
        :param int dimensions: number of dimensions the ensemble represents
        :param tuple max_rate_uniform: distribution of max firing rates of the neurons in the ensemble
        :param tuple intercept_uniform: distribution of neuron intercepts
        :param float radius: radius
        :param list encoders: the encoders
        :returns: the newly created ensemble

        """
        e = SpikingEnsemble(name, num_neurons=num_neurons, dimensions=dimensions,
                              max_rate=nengo.uniform(max_rate[0], max_rate[1]),
                              intercept=nengo.uniform(intercept[0], intercept[1]),
                              radius=radius, encoders=encoders)

        # store created ensemble in node dictionary
        self.add(e)
        return e

    def make_node(self, name, output=None): 
        """Create a Node and add it to the network.

        Output can be either a function or a np.ndarray.
        If it's a function, it should return a np.ndarray.
        It will called like this:
            output() (if it takes no arguments)
            output(simtime) (if it takes one argument)

        Any other arguments your function needs should be retrieved via a closure or
        self or something.
        
        """
        if output == None:
            n = node.Node(name)
        elif callable(output):
            func_args = inspect.getargspec(output).args
            
            print func_args, len(func_args)
            
            if len(func_args) == 1:
                n = node.TimeNode(name, output)
            elif len(func_args) == 0:
                n = node.Node(name, output)
            else:
                print "make_node only accepts output functions with 0 or 1 arguments"
                return None
        elif isinstance(output, basestring):
            pass
        elif isinstance(output, dict):
            n = node.DictNode(name, output)
        elif isinstance(output, (list)):
            n = node.ValueNode(name, output)
        self.add(n)
        return n
        
    def make_network(self, name):
        """Create a subnetwork.     
        
        :param name: the name of the subnetwork to create        
        """
        n = Network(name)
        self.add(n)
        return n

    def probe(self, target, sample_every=0.01, static=False):
        """Add a probe to measure the given target.
        
        :param target: a variable to record
        :param sample_every: the sampling frequency of the probe
        :param static: True if this variable should only be sampled once.
        :returns: The Probe object
        
        """

        target = self.get(target)
        if isinstance(target, SpikingEnsemble):
            target = self.get_object_output(target)
        
        p = probe.ListProbe(
                        target=target,
                        sample_every=sample_every,
                        static=static)

        self.probes.append(p)

    def remove(self, obj):
        """Removes an object from the network.
        
        :param obj: The object to remove
        :param type: <nengo string> or Ensemble, Node, Network, Connection
        """
        pass
