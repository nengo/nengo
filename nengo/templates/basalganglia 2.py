title='Basal Ganglia'
label='Basal\nGanglia'
icon='basalganglia.png'

description="""<html>To drive the constructed basal ganglia, you need an input with dimensions equal to the number of actions. A function input with dimensions equal the number of actions will work. For SPA models inputs are added using the BG Rule template.  </html>"""

params=[
    ('name','Name',str,'Name of the new basal ganglia sub-network'),
    ('dimensions','Number of actions',int,'Number of actions for the basal ganglia to decide between'),
    ('pstc','Input PSTC [s]',float,'Post-synaptic time constant for input filtering, in seconds'),
    ]

def test_params(net,p):
    try:
       net.network.getNode(p['name'])
       return 'That name is already taken'
    except:
        pass
    if p['dimensions']<1: return 'Must have a positive number of actions'
    
import nps.basalganglia
import nef
import numeric
from ca.nengo.model.impl import NetworkImpl
from java.util import ArrayList
from java.util import HashMap
def make(net,name='Basal Ganglia', dimensions=1, neurons=100, pstc=0.01, netbg=None, same_neurons=True, tau_ampa=0.002, tau_gaba=0.008, radius=1.5):

    if netbg is None:
        netbg=nef.Network(name)
    input=netbg.make('input',1,dimensions,quick=True,mode='direct')
    output=netbg.make('output',1,dimensions,quick=True,mode='direct')
    nps.basalganglia.make_basal_ganglia(netbg,input,output, dimensions=dimensions, neurons=neurons, same_neurons=same_neurons, tau_ampa=0.002, tau_gaba=0.008, radius=radius)

    input.addDecodedTermination('input',numeric.eye(dimensions),pstc,False)
    netbg.network.exposeTermination(input.getTermination('input'),'input')
    netbg.network.exposeOrigin(output.getOrigin('X'),'output')
    
    if net is not None:
        net.add(netbg.network)
    
        if net.network.getMetaData("BasalGanglia") == None:
            net.network.setMetaData("BasalGanglia", HashMap())
        bgs = net.network.getMetaData("BasalGanglia")

        bg=HashMap(5)
        bg.put("name", name)
        bg.put("dimensions", dimensions)
        bg.put("neurons", neurons)
        bg.put("pstc", pstc)
        bg.put("same_neurons", same_neurons)

        bgs.put(name, bg)

        if net.network.getMetaData("templates") == None:
            net.network.setMetaData("templates", ArrayList())
        templates = net.network.getMetaData("templates")
        templates.add(name)

    return netbg

