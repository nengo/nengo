title='Decoded Termination'
label='Termination'
icon='termination.png'

from ca.nengo.ui.configurable.descriptors import PTerminationWeights

def params(net,node):
    return [
    ('name','Name',str,'Name of termination'),
    ('weights','Weights',PTerminationWeights('Weights',node.dimension),'Linear transformation matrix to apply'),
    ('pstc','PSTC [s]',float,'Post-synaptic time constant, in seconds'),
    ('modulatory','Is Modulatory',bool,'Only Enable this setting if this termination merely adjusts neuron properties (rather than adding input current)'),
    ]

def test_params(net,node,p):
    try:
        n = node.getTermination(p['name'])
        if n is not None:
            return 'That name is already in use'
    except:
        pass

def test_drop(net,node):
    return hasattr(node,'addDecodedTermination')

def make(net,node,name='termination',weights=None,pstc=0.01,modulatory=False):
    node.addDecodedTermination(name,weights,pstc,modulatory)
