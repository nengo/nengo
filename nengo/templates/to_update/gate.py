title='Gate'
label='Gate'
icon='gate.png'

description="""<html>This template creates an ensemble that drives an inhibitory gate on an existing specified ensemble. </html>"""


params=[
    ('name','Name',str,'Name of the new gating ensemble'),
    ('gated','Name of gated ensemble',str,'Name of the existing ensemble to gate'),
    ('neurons','Number of neurons',int,'Number of neurons in the new gating ensemble'),
    ('pstc','Gating PSTC [s]', float, 'Post-synaptic time constant of the gating ensemble'),
    ]

import nef
import ca.nengo

def test_params(net, p):
    gatedIsSet = False
    nameIsTaken = False
    nodeList = net.network.getNodes()
    for i in nodeList:
        if i.name == p['gated']:
            gatedIsSet = True
        elif i.name == p['name']:
            nameIsTaken = True
    if nameIsTaken: return 'That name is already taken'
    if not gatedIsSet: return 'Must provide the name of an existing ensemble to be gated'
    target=net.network.getNode(p['gated'])
    if not isinstance(target, ca.nengo.model.impl.NetworkArrayImpl) and not isinstance(target, ca.nengo.model.nef.NEFEnsemble):
        return 'The ensemble to be gated must be either an ensemble or a network array'
    if p['neurons']<1: return 'The number of neurons must be greater than zero'
    if p['pstc']<=0: return 'The post-synaptic time constant must be greater than zero'
    
from java.util import ArrayList
from java.util import HashMap

from ca.nengo.model.impl import NetworkArrayImpl
def make(net,name='Gate', gated='visual', neurons=40 ,pstc=0.01):
    gate=net.make(name, neurons, 1, intercept=(-0.7, 0), encoders=[[-1]])
    def addOne(x):
        return [x[0]+1]            
    net.connect(gate, None, func=addOne, origin_name='xBiased', create_projection=False)
    output=net.network.getNode(gated)
    if isinstance(output,NetworkArrayImpl):
        weights=[[-10]]*(output.nodes[0].neurons*len(output.nodes))
    else:
        weights=[[-10]]*output.neurons
    
    count=0
    while 'gate_%02d'%count in [t.name for t in output.terminations]:
        count=count+1
    oname = str('gate_%02d'%count)
    output.addTermination(oname, weights, pstc, False)
    
    orig = gate.getOrigin('xBiased')
    term = output.getTermination(oname)
    net.network.addProjection(orig, term)

    if net.network.getMetaData("gate") == None:
        net.network.setMetaData("gate", HashMap())
    gates = net.network.getMetaData("gate")

    gate=HashMap(4)
    gate.put("name", name)
    gate.put("gated", gated)
    gate.put("neurons", neurons)
    gate.put("pstc", pstc)

    gates.put(name, gate)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(name)

    if net.network.getMetaData("templateProjections") == None:
        net.network.setMetaData("templateProjections", HashMap())
    templateproj = net.network.getMetaData("templateProjections")
    templateproj.put(name, gated)

