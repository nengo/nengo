import warnings

import numpy as np

import nengo
from nengo.config import ConfigParam
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import Default, IntParam, NumberParam
from nengo.spa.module import Module
from nengo.synapses import Lowpass, SynapseParam


# connection weights from (Gurney, Prescott, & Redgrave, 2001)
class Weights(object):
    mm = 1
    mp = 1
    me = 1
    mg = 1
    ws = 1
    wt = 1
    wm = 1
    wg = 1
    wp = 0.9
    we = 0.3
    e = 0.2
    ep = -0.25
    ee = -0.2
    eg = -0.2
    le = 0.2
    lg = 0.2

    @classmethod
    def str_func(cls, x):
        if x < cls.e:
            return 0
        return cls.mm * (x - cls.e)

    @classmethod
    def stn_func(cls, x):
        if x < cls.ep:
            return 0
        return cls.mp * (x - cls.ep)

    @classmethod
    def gpe_func(cls, x):
        if x < cls.ee:
            return 0
        return cls.me * (x - cls.ee)

    @classmethod
    def gpi_func(cls, x):
        if x < cls.eg:
            return 0
        return cls.mg * (x - cls.eg)


class BasalGanglia(Module):
    """Winner take all network, typically used for action selection.

    The basal ganglia network outputs approximately 0 at the dimension with
    the largest value, and is negative elsewhere.

    While the basal ganglia is primarily defined by its winner-take-all
    function, it is also organized to match the organization of the human
    basal ganglia. It consists of five ensembles:

    * Striatal D1 dopamine-receptor neurons (``strD1``)
    * Striatal D2 dopamine-receptor neurons (``strD2``)
    * Subthalamic nucleus (``stn``)
    * Globus pallidus internus / substantia nigra reticulata (``gpi``)
    * Globus pallidus externus (``gpe``)

    Interconnections between these areas are also based on known
    neuroanatomical connections. See [1]_ for more details, and [2]_ for
    the original non-spiking basal ganglia model by
    Gurney, Prescott & Redgrave that this model is based on.

    .. note:: The default `.Solver` for the basal ganglia is `.NnlsL2nz`, which
              requires SciPy. If SciPy is not installed, the global default
              solver will be used instead.

    Parameters
    ----------
    action_count : int
        Number of actions.
    n_neuron_per_ensemble : int, optional (Default: 100)
        Number of neurons in each ensemble in the network.
    output_weight : float, optional (Default: -3.)
        A scaling factor on the output of the basal ganglia
        (specifically on the connection out of the GPi).
    input_bias : float, optional (Default: 0.)
        An amount by which to bias all dimensions of the input node.
        Biasing the input node is important for ensuring that all input
        dimensions are positive and easily comparable.
    general_config : config, optional (Default: None)
        General configuration to be used when specific parameters are not
        overwritten with `ampa_config` or `gaba_config`.
    ampa_config : config, optional (Default: None)
        Configuration for connections corresponding to biological connections
        to AMPA receptors (i.e., connections from STN to to GPi and GPe).
        If None, a default configuration using a 2 ms lowpass synapse
        will be used.
    gaba_config : config, optional (Default: None)
        Configuration for connections corresponding to biological connections
        to GABA receptors (i.e., connections from StrD1 to GPi, StrD2 to GPe,
        and GPe to GPi and STN). If None, a default configuration using an
        8 ms lowpass synapse will be used.
    kwargs
        Passed through the ``spa.Module``.

    Attributes
    ----------
    bias_input : Node or None
        If ``input_bias`` is non-zero, this node will be created to bias
        all of the dimensions of the input signal.
    gpe : EnsembleArray
        Globus pallidus externus ensembles.
    gpi : EnsembleArray
        Globus pallidus internus ensembles.
    input : Node
        Accepts the input signal.
    output : Node
        Provides the output signal.
    stn : EnsembleArray
        Subthalamic nucleus ensembles.
    strD1 : EnsembleArray
        Striatal D1 ensembles.
    strD2 : EnsembleArray
        Striatal D2 ensembles.

    References
    ----------
    .. [1] Stewart, T. C., Choo, X., & Eliasmith, C. (2010).
       Dynamic behaviour of a spiking model of action selection in the
       basal ganglia. In Proceedings of the 10th international conference on
       cognitive modeling (pp. 235-40).
    .. [2] Gurney, K., Prescott, T., & Redgrave, P. (2001).
       A computational model of action selection in the basal
       ganglia. Biological Cybernetics 84, 401-423.
    """

    input_synapse = SynapseParam('input_synapse', default=Lowpass(0.002))
    n_neurons_per_ensemble = IntParam(
        'n_neurons_per_ensemble', default=100, low=1, readonly=True)
    output_weight = NumberParam('output_weight', default=-3., readonly=True)
    input_bias = NumberParam('input_bias', default=0., readonly=True)
    general_config = ConfigParam(
        'general_config', default=None, optional=True, readonly=True)
    ampa_config = ConfigParam(
        'ampa_config', default=None, optional=True, readonly=True)
    gaba_config = ConfigParam(
        'gaba_config', default=None, optional=True, readonly=True)

    def __init__(
            self, action_count, n_neurons_per_ensemble=Default,
            output_weight=Default, input_bias=Default, general_config=Default,
            ampa_config=Default, gaba_config=Default, **kwargs):
        super(BasalGanglia, self).__init__(**kwargs)

        self.action_count = action_count
        self.n_neurons_per_ensemble = n_neurons_per_ensemble
        self.output_weight = output_weight
        self.input_bias = input_bias
        self.general_config = general_config
        self.ampa_config = ampa_config
        self.gaba_config = gaba_config

        self.input_connections = {}
        # Affects all ensembles / connections in the BG
        # unless overwritten with general_config
        config = nengo.Config(nengo.Ensemble, nengo.Connection)
        config[nengo.Ensemble].radius = 1.5
        config[nengo.Ensemble].encoders = nengo.dists.Choice([[1]])
        try:
            # Best, if we have SciPy
            config[nengo.Connection].solver = nengo.solvers.NnlsL2nz()
        except ImportError:
            # Warn if we can't use the better decoder solver.
            if (nengo.Connection not in self.general_config.params or
                    'solver' not in self.general_config[nengo.Connection]):
                warnings.warn("SciPy is not installed, so BasalGanglia will "
                              "use the default decoder solver. Installing "
                              "SciPy may improve BasalGanglia performance.")

        gaba_default = nengo.Config(nengo.Connection)
        gaba_default[nengo.Connection].synapse = nengo.Lowpass(0.008)
        ampa_default = nengo.Config(nengo.Connection)
        ampa_default[nengo.Connection].synapse = nengo.Lowpass(0.002)

        ea_params = {'n_neurons': self.n_neurons_per_ensemble,
                     'n_ensembles': self.action_count}

        with self, config, self.general_config:
            self.strD1 = EnsembleArray(
                label="Striatal D1 neurons",
                intercepts=nengo.dists.Uniform(Weights.e, 1), **ea_params)
            self.strD2 = EnsembleArray(
                label="Striatal D2 neurons",
                intercepts=nengo.dists.Uniform(Weights.e, 1), **ea_params)
            self.stn = EnsembleArray(
                label="Subthalamic nucleus",
                intercepts=nengo.dists.Uniform(Weights.ep, 1), **ea_params)
            self.gpi = EnsembleArray(
                label="Globus pallidus internus",
                intercepts=nengo.dists.Uniform(Weights.eg, 1), **ea_params)
            self.gpe = EnsembleArray(
                label="Globus pallidus externus",
                intercepts=nengo.dists.Uniform(Weights.ee, 1), **ea_params)

            self.input = nengo.Node(label="input", size_in=self.action_count)
            self.output = nengo.Node(label="output", size_in=self.action_count)

            # add bias input (BG performs best in the range 0.5--1.5)
            if abs(self.input_bias) > 0.0:
                self.bias_input = nengo.Node(
                    np.ones(self.action_count) * self.input_bias,
                    label="basal ganglia bias")
                nengo.Connection(self.bias_input, self.input)

            # spread the input to StrD1, StrD2, and STN
            nengo.Connection(self.input, self.strD1.input, synapse=None,
                             transform=Weights.ws * (1 + Weights.lg))
            nengo.Connection(self.input, self.strD2.input, synapse=None,
                             transform=Weights.ws * (1 - Weights.le))
            nengo.Connection(self.input, self.stn.input, synapse=None,
                             transform=Weights.wt)

            # connect the striatum to the GPi and GPe (inhibitory)
            strD1_output = self.strD1.add_output('func_str', Weights.str_func)
            strD2_output = self.strD2.add_output('func_str', Weights.str_func)
            with gaba_default, self.gaba_config:
                nengo.Connection(strD1_output, self.gpi.input,
                                 transform=-Weights.wm)
                nengo.Connection(strD2_output, self.gpe.input,
                                 transform=-Weights.wm)

            # connect the STN to GPi and GPe (broad and excitatory)
            tr = Weights.wp * np.ones((self.action_count, self.action_count))
            stn_output = self.stn.add_output('func_stn', Weights.stn_func)
            with ampa_default, self.ampa_config:
                nengo.Connection(stn_output, self.gpi.input, transform=tr)
                nengo.Connection(stn_output, self.gpe.input, transform=tr)

            # connect the GPe to GPi and STN (inhibitory)
            gpe_output = self.gpe.add_output('func_gpe', Weights.gpe_func)
            with gaba_default, self.gaba_config:
                nengo.Connection(
                    gpe_output, self.gpi.input, transform=-Weights.we)
                nengo.Connection(
                    gpe_output, self.stn.input, transform=-Weights.wg)

            # connect GPi to output (inhibitory)
            gpi_output = self.gpi.add_output('func_gpi', Weights.gpi_func)
            nengo.Connection(gpi_output, self.output, synapse=None,
                             transform=self.output_weight)

    def connect_input(self, source, transform, index):
        self.input_connections[index] = nengo.Connection(
            source, self.input[index], transform=transform,
            synapse=self.input_synapse)
