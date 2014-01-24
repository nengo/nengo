try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import logging
import pickle
import os.path

import numpy as np

import nengo

logger = logging.getLogger(__name__)


class Model(object):
    """A model contains a single network and the ability to
    run simulations of that network.

    Model is the first part of the API that modelers
    become familiar with, and it is possible to create
    many of the models that one would want to create simply
    by making a model and calling functions on that model.

    For example, a model that implements a communication channel
    between two ensembles of neurons can be created with::

        import nengo
        model = nengo.Model("Communication channel")
        input = model.make_node("Input", values=[0])
        pre = model.make_ensemble("In", neurons=100, dimensions=1)
        post = model.make_ensemble("Out", neurons=100, dimensions=1)
        model.connect(input, pre)
        model.connect(pre, post)

    Parameters
    ----------
    name : str
        Name of the model.
    seed : int, optional
        Random number seed that will be fed to the random number generator.
        Setting this seed makes the creation of the model
        a deterministic process; however, each new ensemble
        in the network advances the random number generator,
        so if the network creation code changes, the entire model changes.

    Attributes
    ----------
    name : str
        Name of the model
    seed : int
        Random seed used by the model.
    time : float
        The amount of time that this model has been simulated.
    metadata : dict
        An editable dictionary that modelers can use to store
        extra information about a network.
    properties : read-only dict
        A collection of basic information about
        a network (e.g., number of neurons, number of synapses, etc.)

    """

    def __init__(self, label="Model", seed=None):
        self.objs = []
        self.probed = OrderedDict()
        self.connections = []
        self.signal_probes = []

        self.label = label + ''  # -- make self.name a string, raise error otw
        self.seed = seed


        #make this the default context
        nengo.context.clear()
        nengo.context.append(self)

    def __str__(self):
        return "Model: " + self.label

    ### I/O

    def save(self, fname, format=None):
        """Save this model to a file.

        So far, Pickle is the only implemented format.

        """
        if format is None:
            format = os.path.splitext(fname)[1]

        with open(fname, 'wb') as f:
            pickle.dump(self, f)
            logger.info("Saved %s successfully.", fname)

    @staticmethod
    def load(self, fname, format=None):
        """Load this model from a file.

        So far, JSON and Pickle are the possible formats.

        """
        # if format is None:
        #     format = os.path.splitext(fname)[1]

        # Default to pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)

        raise IOError("Could not load {}".format(fname))

    ### Model manipulation

    def add(self, obj):
        """Adds a Nengo object to this model.

        This is generally only used for manually created nodes, not ones
        created by calling :func:`nef.Model.make_ensemble()` or
        :func:`nef.Model.make_node()`, as these are automatically added.
        A common usage is with user created subclasses, as in the following::

          node = net.add(MyNode('name'))

        Parameters
        ----------
        obj : Nengo object
            The Nengo object to add.

        Returns
        -------
        obj : Nengo object
            The Nengo object that was added.

        See Also
        --------
        Network.add : The same function for Networks

        """
        try:
            obj.add_to_model(self)
            return obj
        except AttributeError as ae:
            raise TypeError("Error in %s.add_to_model.\n%s" % (obj, ae))

    def remove(self, target):
        """Removes a Nengo object from the model.

        Parameters
        ----------
        target : str, Nengo object
            A string referencing the Nengo object to be removed
            (see `string reference <string_reference.html>`)
            or node or name of the node to be removed.

        Returns
        -------
        target : Nengo object
            The Nengo object removed.

        """
        if not target in self.objs:
            logger.warning("%s is not in model %s.", str(target), self.label)
            return

        self.objs = [o for o in self.objs if o != target]
        logger.info("%s removed.", target)

    def __enter__(self):
        nengo.context.append(self)

    def __exit__(self, exception_type, exception_value, traceback):
        nengo.context.pop()
