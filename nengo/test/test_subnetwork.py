"""Test of Network.make_subnetwork(), which should allow for easy nesting of
collections of ensembles.
"""

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

net=nef.Network('Main')

netA=net.make_subnetwork('A')
netB=net.make_subnetwork('B')

net.make('X',50,1)
netA.make('Y',50,1)
netB.make('Z',50,1)
netB.make('W',50,1)

netB.connect('Z','W')     # connection within a subnetwork
net.connect('X','A.Y')    # connection into a subnetwork
net.connect('A.Y','X')    # connection out of a subnetwork
net.connect('A.Y','B.Z')  # connection across subnetworks

netC=netA.make_subnetwork('C')
netC.make('I',50,1)
netC.make('J',50,1)
netC.connect('I','J')       # connection within a subsubnetwork
net.connect('X','A.C.I')    # connection into a subsubnetwork
net.connect('A.C.J','X')    # connection out of a subsubnetwork
net.connect('A.C.J','B.Z')  # connection across subsubnetworks
netA.connect('Y','C.J')     # connection across subnetworks

net.run(1) # run for 1 second
