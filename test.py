# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:03:00 2015

@author: Paxon
"""

import nengo

model=nengo.Network()
with model:
    a = nengo.Ensemble(10,1)
    nengo.Connection(a.neurons,a, synapse=0.1, transform=[[1]*10])
    
sim=nengo.Simulator(model)
