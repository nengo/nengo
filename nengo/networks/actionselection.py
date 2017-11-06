import warnings

import numpy as np

import nengo
from nengo.dists import Choice, Uniform
from nengo.exceptions import ObsoleteError
from nengo.networks.ensemblearray import EnsembleArray
from nengo.solvers import NnlsL2nz


# connection weights from (Gurney, Prescott, & Redgrave, 2001)
class Weights:
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


def config_with_default_synapse(config, synapse):
    if config is None:
        config = nengo.Config(nengo.Connection)
        config[nengo.Connection].synapse = synapse
    override = 'synapse' not in config[nengo.Connection]
    if override:
        config[nengo.Connection].synapse = synapse
    return config, override


class BasalGanglia(nengo.Network):
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
    dimensions : int
        Number of dimensions (i.e., actions).
    n_neurons_per_ensemble : int, optional
        Number of neurons in each ensemble in the network.
    output_weight : float, optional
        A scaling factor on the output of the basal ganglia
        (specifically on the connection out of the GPi).
    input_bias : float, optional
        An amount by which to bias all dimensions of the input node.
        Biasing the input node is important for ensuring that all input
        dimensions are positive and easily comparable.
    ampa_config : config, optional
        Configuration for connections corresponding to biological connections
        to AMPA receptors (i.e., connections from STN to to GPi and GPe).
        If None, a default configuration using a 2 ms lowpass synapse
        will be used.
    gaba_config : config, optional
        Configuration for connections corresponding to biological connections
        to GABA receptors (i.e., connections from StrD1 to GPi, StrD2 to GPe,
        and GPe to GPi and STN). If None, a default configuration using an
        8 ms lowpass synapse will be used.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

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
    def __init__(self,
                 dimensions,
                 n_neurons_per_ensemble=100,
                 output_weight=-3.,
                 input_bias=0.,
                 ampa_config=None,
                 gaba_config=None,
                 **kwargs):
        if 'net' in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault('label', "Basal Ganglia")
        super().__init__(**kwargs)

        ampa_config, override_ampa = config_with_default_synapse(
            ampa_config, nengo.Lowpass(0.002))
        gaba_config, override_gaba = config_with_default_synapse(
            gaba_config, nengo.Lowpass(0.008))

        # Affects all ensembles / connections in the BG
        # unless they've been overridden on `self.config`
        config = nengo.Config(nengo.Ensemble, nengo.Connection)
        config[nengo.Ensemble].radius = 1.5
        config[nengo.Ensemble].encoders = Choice([[1]])
        try:
            # Best, if we have SciPy
            config[nengo.Connection].solver = NnlsL2nz()
        except ImportError:
            # Warn if we can't use the better decoder solver.
            warnings.warn("SciPy is not installed, so BasalGanglia will "
                          "use the default decoder solver. Installing SciPy "
                          "may improve BasalGanglia performance.")

        ea_params = {'n_neurons': n_neurons_per_ensemble,
                     'n_ensembles': dimensions}

        with self, config:
            self.strD1 = EnsembleArray(label="Striatal D1 neurons",
                                       intercepts=Uniform(Weights.e, 1),
                                       **ea_params)
            self.strD2 = EnsembleArray(label="Striatal D2 neurons",
                                       intercepts=Uniform(Weights.e, 1),
                                       **ea_params)
            self.stn = EnsembleArray(label="Subthalamic nucleus",
                                     intercepts=Uniform(Weights.ep, 1),
                                     **ea_params)
            self.gpi = EnsembleArray(label="Globus pallidus internus",
                                     intercepts=Uniform(Weights.eg, 1),
                                     **ea_params)
            self.gpe = EnsembleArray(label="Globus pallidus externus",
                                     intercepts=Uniform(Weights.ee, 1),
                                     **ea_params)

            self.input = nengo.Node(label="input", size_in=dimensions)
            self.output = nengo.Node(label="output", size_in=dimensions)

            # add bias input (BG performs best in the range 0.5--1.5)
            if abs(input_bias) > 0.0:
                self.bias_input = nengo.Node(np.ones(dimensions) * input_bias,
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
            with gaba_config:
                nengo.Connection(strD1_output, self.gpi.input,
                                 transform=-Weights.wm)
                nengo.Connection(strD2_output, self.gpe.input,
                                 transform=-Weights.wm)

            # connect the STN to GPi and GPe (broad and excitatory)
            tr = Weights.wp * np.ones((dimensions, dimensions))
            stn_output = self.stn.add_output('func_stn', Weights.stn_func)
            with ampa_config:
                nengo.Connection(stn_output, self.gpi.input, transform=tr)
                nengo.Connection(stn_output, self.gpe.input, transform=tr)

            # connect the GPe to GPi and STN (inhibitory)
            gpe_output = self.gpe.add_output('func_gpe', Weights.gpe_func)
            with gaba_config:
                nengo.Connection(
                    gpe_output, self.gpi.input, transform=-Weights.we)
                nengo.Connection(
                    gpe_output, self.stn.input, transform=-Weights.wg)

            # connect GPi to output (inhibitory)
            gpi_output = self.gpi.add_output('func_gpi', Weights.gpi_func)
            nengo.Connection(
                gpi_output, self.output, synapse=None, transform=output_weight)

        # Return ampa_config and gaba_config to previous states, if changed
        if override_ampa:
            del ampa_config[nengo.Connection].synapse
        if override_gaba:
            del gaba_config[nengo.Connection].synapse


class Thalamus(nengo.Network):
    """Inhibits non-selected actions.

    The thalamus is intended to work in tandem with a basal ganglia network.
    It converts basal ganglia output into a signal with (approximately) 1 for
    the selected action and 0 elsewhere.

    In order to suppress low responses and strengthen high responses,
    a constant bias is added to each dimension (i.e., action), and dimensions
    mutually inhibit each other. Additionally, the ensemble representing
    each dimension is created with positive encoders and can be assigned
    positive x-intercepts to threshold low responses.

    Parameters
    ----------
    dimensions : int
        Number of dimensions (i.e., actions).
    n_neurons_per_ensemble : int, optional
        Number of neurons in each ensemble in the network.
    mutual_inhib : float, optional
        Strength of the mutual inhibition between actions.
    threshold : float, optional
        The threshold below which values will not be represented.
    **kwargs
        Keyword arguments passed through to ``nengo.Network``
        like 'label' and 'seed'.

    Attributes
    ----------
    actions : EnsembleArray
        Each ensemble represents one dimension (action).
    bias : Node
        The constant bias injected in each ``actions`` ensemble.
    input : Node
        Input to the ``actions`` ensembles.
    output : Node
        Output from the ``actions`` ensembles.
    """
    def __init__(self,
                 dimensions,
                 n_neurons_per_ensemble=50,
                 mutual_inhib=1.,
                 threshold=0.,
                 **kwargs):
        if 'net' in kwargs:
            raise ObsoleteError("The 'net' argument is no longer supported.")
        kwargs.setdefault('label', "Thalamus")
        super().__init__(**kwargs)

        with self:
            self.actions = EnsembleArray(n_neurons_per_ensemble, dimensions,
                                         intercepts=Uniform(threshold, 1),
                                         encoders=Choice([[1.0]]),
                                         label="actions")
            nengo.Connection(self.actions.output, self.actions.input,
                             transform=(np.eye(dimensions) - 1) * mutual_inhib)
            self.bias = nengo.Node([1], label="thalamus bias")
            nengo.Connection(self.bias, self.actions.input,
                             transform=np.ones((dimensions, 1)))

        self.input = self.actions.input
        self.output = self.actions.output
