from .. import objects
from . import Network
from ..templates import EnsembleArray

import nengo
import numpy as np

class BasalGanglia(Network):
    def make(self, dimensions, neurons_per_ensemble=100, radius=1.5,
                    tau_ampa=0.002, tau_gaba=0.008, output_weight=-3):
                    
        # connection weights from (Gurney, Prescott, & Redgrave, 2001)            
        mm=1; mp=1; me=1; mg=1
        ws=1; wt=1; wm=1; wg=1; wp=0.9; we=0.3
        e=0.2; ep=-0.25; ee=-0.2; eg=-0.2
        le=0.2; lg=0.2
        
        encoders = [[1.0]]*neurons_per_ensemble
        strD1 = self.add(EnsembleArray('StrD1', 
                        nengo.LIF(neurons_per_ensemble*dimensions), 
                        dimensions, radius=radius, 
                        intercepts=objects.Uniform(e,1)))
        for ens in strD1.ensembles:
            ens.encoders = encoders
            
        strD2 = self.add(EnsembleArray('StrD2', 
                        nengo.LIF(neurons_per_ensemble*dimensions), 
                        dimensions, radius=radius, 
                        intercepts=objects.Uniform(e,1)))
        for ens in strD2.ensembles:
            ens.encoders = encoders
         
        stn = self.add(EnsembleArray('STN', 
                        nengo.LIF(neurons_per_ensemble*dimensions), 
                        dimensions, radius=radius, 
                        intercepts=objects.Uniform(ep,1)))
        for ens in stn.ensembles:
            ens.encoders = encoders
                    
        gpi = self.add(EnsembleArray('GPi', 
                        nengo.LIF(neurons_per_ensemble*dimensions), 
                        dimensions, radius=radius, 
                        intercepts=objects.Uniform(eg,1)))
        for ens in gpi.ensembles:
            ens.encoders = encoders

        gpe = self.add(EnsembleArray('GPe', 
                        nengo.LIF(neurons_per_ensemble*dimensions), 
                        dimensions, radius=radius, 
                        intercepts=objects.Uniform(ee,1)))
        for ens in gpe.ensembles:
            ens.encoders = encoders
            
        self.input = self.add(
                    objects.PassthroughNode("Input", dimensions=dimensions))
        self.output = self.add(
                    objects.PassthroughNode("Output", dimensions=dimensions))
        
        # spread the input to StrD1, StrD2, and STN
        self.input.connect_to(strD1, transform=np.eye(dimensions)*ws*(1+lg), filter=None)
        self.input.connect_to(strD2, transform=np.eye(dimensions)*ws*(1-le), filter=None)
        self.input.connect_to(stn, transform=np.eye(dimensions)*wt, filter=None)
        
        # connect the striatum to the GPi and GPe (inhibitory)
        def func_str(x):
            if x[0] < e: return 0
            return mm * (x[0] - e)
        strD1.connect_to(gpi, function=func_str, 
                        transform=-np.eye(dimensions)*wm, filter=tau_gaba)
        strD2.connect_to(gpe, function=func_str,
                        transform=-np.eye(dimensions)*wm, filter=tau_gaba)

                        
        # connect the STN to GPi and GPe (broad and excitatory)
        def func_stn(x):
            if x[0] < ep: return 0
            return mp * (x[0] - ep)
        tr = [[wp] * dimensions for i in range(dimensions)]    
        stn.connect_to(gpi, function=func_stn, transform=tr, filter=tau_ampa)
        stn.connect_to(gpe, function=func_stn, transform=tr, filter=tau_ampa)        

        # connect the GPe to GPi and STN (inhibitory)
        def func_gpe(x):
            if x[0] < ee: return 0
            return me * (x[0] - ee)
        gpe.connect_to(gpi, function=func_gpe, filter=tau_gaba,
            transform=-np.eye(dimensions)*we)
        gpe.connect_to(stn, function=func_gpe, pstc=tau_gaba,
            transform=-np.eye(dimensions)*wg)

        #connect GPi to output (inhibitory)
        def func_gpi(x):
            if x[0]<eg: return 0
            return mg*(x[0]-eg)
        gpi.connect_to(self.output, function=func_gpi, filter=None, 
            transform=np.eye(dimensions)*output_weight)        
            
