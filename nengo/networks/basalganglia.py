import numpy as np

import nengo
from nengo.decoders import nnls_L2nz
from nengo.objects import Uniform
from nengo.networks.ensemblearray import EnsembleArray


class BasalGanglia(nengo.Network):
    """Winner takes all; outputs 0 at max dimension, negative elsewhere."""

    # connection weights from (Gurney, Prescott, & Redgrave, 2001)
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

    def __init__(self, dimensions, n_neurons_per_ensemble=100, radius=1.5,
                 tau_ampa=0.002, tau_gaba=0.008, output_weight=-3,
                 decoder_solver=nnls_L2nz):
        encoders = np.ones((n_neurons_per_ensemble, 1))
        ea_params = {
            'neurons': nengo.LIF(n_neurons_per_ensemble),
            'n_ensembles': dimensions,
            'radius': radius,
            'encoders': encoders,
        }

        strD1 = EnsembleArray(label="Striatal D1 neurons",
                              intercepts=Uniform(self.e, 1), **ea_params)

        strD2 = EnsembleArray(label="Striatal D2 neurons",
                              intercepts=Uniform(self.e, 1), **ea_params)

        stn = EnsembleArray(label="Subthalamic nucleus",
                            intercepts=Uniform(self.ep, 1), **ea_params)

        gpi = EnsembleArray(label="Globus pallidus internus",
                            intercepts=Uniform(self.eg, 1), **ea_params)

        gpe = EnsembleArray(label="Globus pallidus externus",
                            intercepts=Uniform(self.ee, 1), **ea_params)

        self.input = nengo.Node(label="input", size_in=dimensions)
        self.output = nengo.Node(label="output", size_in=dimensions)

        # spread the input to StrD1, StrD2, and STN
        nengo.Connection(self.input, strD1.input, synapse=None,
                         transform=self.ws * (1 + self.lg))
        nengo.Connection(self.input, strD2.input, synapse=None,
                         transform=self.ws * (1 - self.le))
        nengo.Connection(self.input, stn.input, synapse=None,
                         transform=self.wt)

        # connect the striatum to the GPi and GPe (inhibitory)
        strD1_output = strD1.add_output(
            'func_str', self.str, decoder_solver=decoder_solver)
        nengo.Connection(strD1_output,
                         gpi.input, synapse=tau_gaba,
                         transform=-np.eye(dimensions) * self.wm)
        strD2_output = strD2.add_output(
            'func_str', self.str, decoder_solver=decoder_solver)
        nengo.Connection(strD2_output,
                         gpe.input, synapse=tau_gaba,
                         transform=-np.eye(dimensions) * self.wm)

        # connect the STN to GPi and GPe (broad and excitatory)
        tr = self.wp * np.ones((dimensions, dimensions))
        stn_output = stn.add_output(
            'func_stn', self.stn, decoder_solver=decoder_solver)
        nengo.Connection(stn_output, gpi.input,
                         transform=tr, synapse=tau_ampa)
        nengo.Connection(stn_output, gpe.input,
                         transform=tr, synapse=tau_ampa)

        # connect the GPe to GPi and STN (inhibitory)
        gpe_output = gpe.add_output(
            'func_gpe', self.gpe, decoder_solver=decoder_solver)
        nengo.Connection(gpe_output, gpi.input, synapse=tau_gaba,
                         transform=-self.we)
        nengo.Connection(gpe_output, stn.input, synapse=tau_gaba,
                         transform=-self.wg)

        # connect GPi to output (inhibitory)
        gpi_output = gpi.add_output(
            'func_gpi', self.gpi, decoder_solver=decoder_solver)
        nengo.Connection(gpi_output, self.output, synapse=None,
                         transform=output_weight)

    @classmethod
    def str(cls, x):
        if x < cls.e:
            return 0
        return cls.mm * (x - cls.e)

    @classmethod
    def stn(cls, x):
        if x < cls.ep:
            return 0
        return cls.mp * (x - cls.ep)

    @classmethod
    def gpe(cls, x):
        if x < cls.ee:
            return 0
        return cls.me * (x - cls.ee)

    @classmethod
    def gpi(cls, x):
        if x < cls.eg:
            return 0
        return cls.mg * (x - cls.eg)
