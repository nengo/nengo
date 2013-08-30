title='Decoded Origin'
label='Origin'
icon='origin.png'
 
from ca.nengo.ui.configurable.descriptors import PFunctionArray
from ca.nengo.model.impl import NetworkArrayImpl

def params(net,node):
    if(isinstance(node, NetworkArrayImpl)):
      dimension = node.getNodes()[0].getDimension();
      function_title = 'Function Array (per sub-ensemble)'
      help_string = """Functions to be computed by each sub-ensemble. The input dimension of each function can 
      be no greater than the dimension of the sub-ensembles. The results from each sub-ensemble are concatenated 
      into one vector, which is the result of this origin. Thus, the dimension of this origin is number of 
      sub-ensembles x number of functions specified.\n"""
      help_string = 'you\n'
      

    else:
      dimension = node.getDimension();
      function_title = "Function Array"
      help_string = "Functions to be computed by the ensemble. The dimension of the resulting origin equals the number of functions specified"

    return [
    ('name','Name',str,'Name of origin'),
    ('functions','Output Functions', PFunctionArray(function_title, dimension), help_string)
    ]

def test_params(net,node,p):
    try:
        n = node.getOrigin(p['name'])
        if n is not None:
            return 'That name is already in use'
    except:
        pass

def test_drop(net,node):
    return hasattr(node,'addDecodedOrigin')

def make(net,node,name='origin', functions=[]):
    node.addDecodedOrigin(name,functions,"AXON")
