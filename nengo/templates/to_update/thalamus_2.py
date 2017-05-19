import nef

title='Thalamus'
label='Thalamus'
icon='thalamus.png'

description="""<html>This template is designed to take output from a Basal Ganglia template model and send it to cortex for control of information flow or update of cortical states in SPA models.</html>"""

params=[
    ('name','Name',str,'Name of the new thalamus sub-network'),
    ('neurons','Neurons per dimension',int,'Number of neurons for each dimension of the thalamus'),
    ('dimensions','Dimensions',int,'Number of dimensions for the thalamus'),
    ('useQuick', 'Quick mode', bool,'Whether to use quick mode'),
    ]

def test_params(net,p):
    try:
       net.network.getNode(p['name'])
       return 'That name is already taken'
    except:
        pass

from java.util import ArrayList
from java.util import HashMap
def make(net,name='Network Array', neurons=50, dimensions=2, inhib_scale=3, tau_inhib=.005, useQuick=True):
    thalamus = net.make_array(name, neurons, dimensions, max_rate=(100,300), intercept=(-1, 0), radius=1, encoders=[[1]], quick=useQuick)    

    # setup inhibitory scaling matrix
    inhib_scaling_matrix = [[0]*dimensions for i in range(dimensions)]
    for i in range(dimensions):
        inhib_scaling_matrix[i][i] = -inhib_scale
    # setup inhibitory matrix
    inhib_matrix = []
    for i in range(dimensions):
        inhib_matrix_part = [[inhib_scaling_matrix[i]] * neurons]
        inhib_matrix.append(inhib_matrix_part[0])

    thalamus.addTermination('bg_input', inhib_matrix, tau_inhib, False)

    def addOne(x):
        return [x[0]+1]            
    net.connect(thalamus, None, func=addOne, origin_name='xBiased', create_projection=False)
    
    if net.network.getMetaData("Thalamus") == None:
        net.network.setMetaData("Thalamus", HashMap())
    thals = net.network.getMetaData("Thalamus")

    thal=HashMap(6)
    thal.put("name", name)
    thal.put("neurons", neurons)
    thal.put("dimensions", dimensions)
    thal.put("inhib_scale", inhib_scale)
    thal.put("tau_inhib", tau_inhib)
    thal.put("useQuick", useQuick)

    thals.put(name, thal)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(name)

    return thalamus
