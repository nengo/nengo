import numpy as np

import nengo
from nengo.objects import Uniform
from .ensemblearray import EnsembleArray


class BasalGanglia(object):
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
                 label='Basal Ganglia'):

        self.label = label
        encoders = np.ones((n_neurons_per_ensemble, 1))
        ea_params = {
            'neurons': nengo.LIF(n_neurons_per_ensemble),
            'n_ensembles': dimensions,
            'radius': radius,
            'encoders': encoders,
        }

        self.strD1 = EnsembleArray(label=label + '.Striatal D1 neurons',
                                   intercepts=Uniform(self.e, 1), **ea_params)

        self.strD2 = EnsembleArray(label=label + '.Striatal D2 neurons',
                                   intercepts=Uniform(self.e, 1), **ea_params)

        self.stn = EnsembleArray(label=label + '.Subthalamic nucleus',
                                 intercepts=Uniform(self.ep, 1), **ea_params)

        self.gpi = EnsembleArray(label=label + '.Globus pallidus internus',
                                 intercepts=Uniform(self.eg, 1), **ea_params)

        self.gpe = EnsembleArray(label=label + '.Globus pallidus externus',
                                 intercepts=Uniform(self.ee, 1), **ea_params)

        self.input = nengo.Node(label=label + ".input", size_in=dimensions)
        self.output = nengo.Node(label=label + ".output", size_in=dimensions)

        # spread the input to StrD1, StrD2, and STN
        nengo.Connection(
            self.input, self.strD1.input, filter=None,
            transform=np.eye(dimensions) * self.ws * (1 + self.lg))
        nengo.Connection(
            self.input, self.strD2.input, filter=None,
            transform=np.eye(dimensions) * self.ws * (1 - self.le))
        nengo.Connection(
            self.input, self.stn.input, filter=None,
            transform=np.eye(dimensions) * self.wt)

        # connect the striatum to the GPi and GPe (inhibitory)
        nengo.Connection(self.strD1.add_output('func_str', self.str_f()),
                         self.gpi.input, filter=tau_gaba,
                         transform=-np.eye(dimensions) * self.wm)
        nengo.Connection(self.strD2.add_output('func_str', self.str_f()),
                         self.gpe.input, filter=tau_gaba,
                         transform=-np.eye(dimensions) * self.wm)

        # connect the STN to GPi and GPe (broad and excitatory)
        tr = np.ones((dimensions, dimensions)) * self.wp
        stn_output = self.stn.add_output('func_stn', self.stn_f())
        nengo.Connection(stn_output, self.gpi.input,
                         transform=tr, filter=tau_ampa)
        nengo.Connection(stn_output, self.gpe.input,
                         transform=tr, filter=tau_ampa)

        # connect the GPe to GPi and STN (inhibitory)
        gpe_output = self.gpe.add_output('func_gpe', self.gpe_f())
        nengo.Connection(gpe_output, self.gpi.input, filter=tau_gaba,
                         transform=-np.eye(dimensions) * self.we)
        nengo.Connection(gpe_output, self.stn.input, filter=tau_gaba,
                         transform=-np.eye(dimensions) * self.wg)

        #connect GPi to output (inhibitory)
        nengo.Connection(self.gpi.add_output('func_gpi', self.gpi_f()),
                         self.output, filter=None,
                         transform=np.eye(dimensions) * output_weight)

    def str_f(self):
        def func_str(x):
            if x[0] < self.e:
                return 0
            return self.mm * (x[0] - self.e)
        return func_str

    def stn_f(self):
        def func_stn(x):
            if x[0] < self.ep:
                return 0
            return self.mp * (x[0] - self.ep)
        return func_stn

    def gpe_f(self):
        def func_gpe(x):
            if x[0] < self.ee:
                return 0
            return self.me * (x[0] - self.ee)
        return func_gpe

    def gpi_f(self):
        def func_gpi(x):
            if x[0] < self.eg:
                return 0
            return self.mg * (x[0] - self.eg)
        return func_gpi
