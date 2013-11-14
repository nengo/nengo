from .. import objects
from . import Network
from ..templates import EnsembleArray

import nengo
import numpy as np

class BasalGanglia(Network):
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

    def make(self, dimensions, n_neurons_per_ensemble=100, radius=1.5,
             tau_ampa=0.002, tau_gaba=0.008, output_weight=-3):

        encoders = np.ones((n_neurons_per_ensemble, 1))
        ea_params = {
            'neurons': nengo.LIF(n_neurons_per_ensemble * dimensions),
            'n_ensembles': dimensions,
            'radius': radius,
            'encoders': encoders,
        }

        strD1 = self.add(EnsembleArray(
            'StrD1', intercepts=objects.Uniform(self.e, 1), **ea_params))

        strD2 = self.add(EnsembleArray(
            'StrD2', intercepts=objects.Uniform(self.e, 1), **ea_params))

        stn = self.add(EnsembleArray(
            'STN', intercepts=objects.Uniform(self.ep, 1), **ea_params))

        gpi = self.add(EnsembleArray(
            'GPi', intercepts=objects.Uniform(self.eg, 1), **ea_params))

        gpe = self.add(EnsembleArray(
            'GPe', intercepts=objects.Uniform(self.ee, 1), **ea_params))

        self.input = self.add(
            objects.Node("input", dimensions=dimensions))
        self.output = self.add(
            objects.Node("output", dimensions=dimensions))

        # spread the input to StrD1, StrD2, and STN
        self.input.connect_to(strD1, filter=None,
            transform=np.eye(dimensions) * self.ws * (1 + self.lg))
        self.input.connect_to(strD2, filter=None,
            transform=np.eye(dimensions) * self.ws * (1 - self.le))
        self.input.connect_to(stn, filter=None,
            transform=np.eye(dimensions) * self.wt)

        # connect the striatum to the GPi and GPe (inhibitory)
        def func_str(x):
            if x[0] < self.e:
                return 0
            return self.mm * (x[0] - self.e)
        strD1.connect_to(gpi, function=func_str, filter=tau_gaba,
                         transform=-np.eye(dimensions) * self.wm)
        strD2.connect_to(gpe, function=func_str, filter=tau_gaba,
                        transform=-np.eye(dimensions) * self.wm)

        # connect the STN to GPi and GPe (broad and excitatory)
        def func_stn(x):
            if x[0] < self.ep:
                return 0
            return self.mp * (x[0] - self.ep)
        tr = np.ones((dimensions, dimensions)) * self.wp
        stn.connect_to(gpi, function=func_stn, transform=tr, filter=tau_ampa)
        stn.connect_to(gpe, function=func_stn, transform=tr, filter=tau_ampa)

        # connect the GPe to GPi and STN (inhibitory)
        def func_gpe(x):
            if x[0] < self.ee:
                return 0
            return self.me * (x[0] - self.ee)
        gpe.connect_to(gpi, function=func_gpe, filter=tau_gaba,
                       transform=-np.eye(dimensions) * self.we)
        gpe.connect_to(stn, function=func_gpe, filter=tau_gaba,
                       transform=-np.eye(dimensions) * self.wg)

        #connect GPi to output (inhibitory)
        def func_gpi(x):
            if x[0] < self.eg:
                return 0
            return self.mg * (x[0] - self.eg)
        gpi.connect_to(self.output, function=func_gpi, filter=None,
                       transform=np.eye(dimensions) * output_weight)
