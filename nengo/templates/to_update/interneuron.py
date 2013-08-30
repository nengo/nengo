title='Interneurons'
label='Interneurons'
icon='interneuron.png'

description="""This template creates inhibitory interneurons.  This adjusts the connection weights so that instead of having a mixture of excitatory and inhibitory synapses, the primary connection is all excitatory, and a separate group of neurons is created to provide the inhibition."""

def params(net,node):
    return [
        ('neurons', 'Number of neurons', int, 'How many interneurons to create'),
        ('pstc', 'Post-synaptic time constant (s)', float, 'Time constant for connection from interneurons to target population'),
        ]

def test_params(net, node, p):
    if p['neurons']<=0:
        return 'The number of neurons must be greater than zero'
    if p['pstc']<=0:
        return 'The post-synaptic time constant must be greater than zero'


import ca.nengo

def test_drop(net, node):
    if isinstance(node, ca.nengo.model.nef.impl.DecodedTermination):
        proj=None
        for p in net.projections:
            if p.termination==node:
                if isinstance(p.origin, ca.nengo.model.nef.impl.DecodedOrigin):
                    return True
                break
    return False    

def make(net, node, neurons=50, pstc=0.01):
    if not isinstance(node, ca.nengo.model.nef.impl.DecodedTermination):
        raise Exception('Interneurons must be created on a Decoded Termination')
    proj=None
    for p in net.network.projections:
        if p.termination==node:
            proj=p
            break
    if proj is None:
        raise Exception('Interneurons must be created on an already-existing projection')
    
    if not isinstance(p.origin, ca.nengo.model.nef.impl.DecodedOrigin):
        raise Exception('Interneurons require a projection from a Decoded Origin to a Decoded Termination')
        
    proj.addBias(neurons, pstc, proj.termination.tau, True, False)
    
        
