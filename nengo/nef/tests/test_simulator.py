import numpy as np

from nengo.nef.simulator import Simulator
from nengo.nef.model import Model
from nengo.nef.node import TimeNode

def test_node():
    mdl = Model('a', seed=123)
    tn = TimeNode('sin', np.sin)
    mdl.network.add(tn)
    mdl.network.probe(tn.default_output)
    mdl.build()
    mdl.reset()
    mdl.run(.1)

    #mdl.connect('sin', 'pow')
    assert mdl.network.Nodes[0]
    assert mdl.network.Probes[0]
    for thing in mdl.network.Probes[0].data:
        print thing

    #XXX todo test what the output should be


def test_ensemble():

    mdl = Model('a', seed=123)
    e = mdl.make_ensemble('a', 100, 1 )
    mdl.network.probe(e.default_output)
    mdl.build()
    mdl.reset()
    mdl.run(.1)

    #mdl.connect('sin', 'pow')
    assert mdl.network.Nodes[0]
    assert mdl.network.Probes[0]
    for thing in mdl.network.Probes[0].data:
        print thing

