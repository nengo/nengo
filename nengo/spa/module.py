import numpy as np

import nengo
from nengo.config import Config
from nengo.exceptions import SpaModuleError
from nengo.spa.vocab import VocabularySet, VocabularySetParam
from nengo.utils.compat import iteritems


class Module(nengo.Network):
    """Base class for SPA Modules.

    Modules are networks that also have a list of inputs and outputs,
    each with an associated `.Vocabulary` (or a desired dimensionality for
    the vocabulary).

    The inputs and outputs are dictionaries that map a name to an
    (object, Vocabulary) pair. The object can be a `.Node` or an `.Ensemble`.
    """

    vocabs = VocabularySetParam('vocabs', default=None, optional=False)

    def __init__(
            self, label=None, seed=None, add_to_container=None, vocabs=None):
        super(Module, self).__init__(label, seed, add_to_container)
        self.config.configures(Module)

        if vocabs is None:
            vocabs = Config.default(Module, 'vocabs')
            if vocabs is None:
                if seed is not None:
                    rng = np.random.RandomState(seed)
                else:
                    rng = None
                vocabs = VocabularySet(rng=rng)
        self.vocabs = vocabs
        self.config[Module].vocabs = vocabs

        self._modules = {}

        self.inputs = {}
        self.outputs = {}

    def on_add(self, spa):
        """Called when this is assigned to a variable in the SPA network.

        Overload this when you want processing to be delayed until after
        the module is attached to the SPA network. This is usually for
        modules that connect to other things in the SPA model (such as
        the basal ganglia or thalamus).
        """

    def __setattr__(self, key, value):
        """A setattr that handles Modules being added specially.

        This is so that we can use the variable name for the Module as
        the name that all of the SPA system will use to access that module.
        """
        if hasattr(self, key) and isinstance(getattr(self, key), Module):
            raise SpaModuleError("Cannot re-assign module-attribute %s to %s. "
                                 "SPA module-attributes can only be assigned "
                                 "once." % (key, value))
        super(Module, self).__setattr__(key, value)
        if isinstance(value, Module):
            if value.label is None:
                value.label = key
            self._modules[key] = value
            for k, (obj, v) in iteritems(value.inputs):
                if isinstance(v, int):
                    value.inputs[k] = (obj, self.vocabs.get_or_create(v))
            for k, (obj, v) in iteritems(value.outputs):
                if isinstance(v, int):
                    value.outputs[k] = (obj, self.vocabs.get_or_create(v))

            value.on_add(self)

    def __exit__(self, ex_type, ex_value, traceback):
        super(Module, self).__exit__(ex_type, ex_value, traceback)
        if ex_type is not None:
            # re-raise the exception that triggered this __exit__
            return False

        module_list = frozenset(self._modules.values())
        for net in self.networks:
            # Since there are no attributes to distinguish what's been added
            # and what hasn't, we have to ask the network
            if isinstance(net, Module) and (net not in module_list):
                raise SpaModuleError("%s must be set as an attribute of "
                                     "a SPA network" % (net))

    def get_module(self, name):
        """Return the module for the given name."""
        if name in self._modules:
            return self._modules[name]
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            if module in self._modules:
                return self._modules[module]
        raise SpaModuleError("Could not find module %r" % name)

    def get_module_input(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a module, or of the form
        <module_name>_<input_name>.
        """
        if name in self._modules and 'default' in self._modules[name].inputs:
            return self._modules[name].inputs['default']
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            if module in self._modules:
                m = self._modules[module]
                if name in m.inputs:
                    return m.inputs[name]
        raise SpaModuleError("Could not find module input %r" % name)

    def get_input_vocab(self, name):
        return self.get_module_input(name)[1]

    def get_module_output(self, name):
        """Return the object to connect into for the given name.

        The name will be either the same as a module, or of the form
        <module_name>_<output_name>.
        """
        if name in self._modules:
            return self._modules[name].outputs['default']
        elif '_' in name:
            module, name = name.rsplit('_', 1)
            if module in self._modules:
                m = self._modules[module]
                if name in m.outputs:
                    return m.outputs[name]
        raise SpaModuleError("Could not find module output %r" % name)

    def get_output_vocab(self, name):
        return self.get_module_output(name)[1]

    def similarity(self, data, probe, vocab=None):
        """Return the similarity between the probed data and corresponding
        vocabulary.

        Parameters
        ----------
        data: ProbeDict
            Collection of simulation data returned by sim.run() function call.
        probe: Probe
            Probe with desired data.
        """
        if vocab is None:
            vocab = self.config[probe.target].vocab
        return nengo.spa.similarity(data[probe], vocab)
