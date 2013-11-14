import numpy as np

import nengo
from ..objects import Uniform
from .ensemblearray import EnsembleArray


class BasalGanglia(nengo.Network):
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

        with self:
            strD1 = EnsembleArray(label='Striatal D1 neurons',
                                  intercepts=Uniform(self.e, 1), **ea_params)

            strD2 = EnsembleArray(label='Striatal D2 neurons',
                                  intercepts=Uniform(self.e, 1), **ea_params)

            stn = EnsembleArray(label='Subthalamic nucleus',
                                intercepts=Uniform(self.ep, 1), **ea_params)

            gpi = EnsembleArray(label='Globus pallidus internus',
                                intercepts=Uniform(self.eg, 1), **ea_params)

            gpe = EnsembleArray(label='Globus pallidus externus',
                                intercepts=Uniform(self.ee, 1), **ea_params)

            self.input = nengo.Node(label="input", dimensions=dimensions)
            self.output = nengo.Node(label="output", dimensions=dimensions)

            # spread the input to StrD1, StrD2, and STN
            nengo.Connection(self.input,
                strD1.input, filter=None,
                transform=np.eye(dimensions) * self.ws * (1 + self.lg))
            nengo.Connection(self.input,
                strD2.input, filter=None,
                transform=np.eye(dimensions) * self.ws * (1 - self.le))
            nengo.Connection(self.input,
                stn.input, filter=None,
                transform=np.eye(dimensions) * self.wt)

            # connect the striatum to the GPi and GPe (inhibitory)
            def func_str(x):
                if x[0] < self.e:
                    return 0
                return self.mm * (x[0] - self.e)
            nengo.Connection(strD1.add_output('func_str', func_str),
                             gpi.input, filter=tau_gaba,
                             transform=-np.eye(dimensions) * self.wm)
            nengo.Connection(strD2.add_output('func_str', func_str),
                             gpe.input, filter=tau_gaba,
                             transform=-np.eye(dimensions) * self.wm)

            # connect the STN to GPi and GPe (broad and excitatory)
            def func_stn(x):
                if x[0] < self.ep:
                    return 0
                return self.mp * (x[0] - self.ep)
            tr = np.ones((dimensions, dimensions)) * self.wp
            stn_output = stn.add_output('func_stn', func_stn)
            nengo.Connection(stn_output, gpi.input,
                             transform=tr, filter=tau_ampa)
            nengo.Connection(stn_output, gpe.input,
                             transform=tr, filter=tau_ampa)

            # connect the GPe to GPi and STN (inhibitory)
            def func_gpe(x):
                if x[0] < self.ee:
                    return 0
                return self.me * (x[0] - self.ee)
            gpe_output = gpe.add_output('func_gpe', func_gpe)
            nengo.Connection(gpe_output, gpi.input, filter=tau_gaba,
                             transform=-np.eye(dimensions) * self.we)
            nengo.Connection(gpe_output, stn.input, filter=tau_gaba,
                             transform=-np.eye(dimensions) * self.wg)

            #connect GPi to output (inhibitory)
            def func_gpi(x):
                if x[0] < self.eg:
                    return 0
                return self.mg * (x[0] - self.eg)
            nengo.Connection(gpi.add_output('func_gpi', func_gpi),
                             self.output, filter=None,
                             transform=np.eye(dimensions) * output_weight)
