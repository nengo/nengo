title='Learned Termination'
label='Learned\nTermination'
icon='learn.png'

description="""<html>This template introduces error-modulated learning into a pre-existing connection in the network. The presynaptic side is 'pre' and the postsynaptic side is 'post'. The error signal must be computed in the added error ensemble, and is usually a gradient.</html>"""

params=[
    ('errName','Name of (new) error ensemble',str,
      "Provides modulatory input to the post ensemble, "
      "modifying its connection from the pre ensemble."
      "<br />Automatically created."),
    ('N_err', 'Number of neurons in error ensemble',int, 'Set the number of neurons in the error ensemble'),
    ('preName','Name of pre ensemble',str, 'Select the name of the pre ensemble (this ensemble must already exist)'),
    ('postName','Name of post ensemble',str,'Select the name of the post ensemble (this ensemble must already exist)'),
    ('rate','Learning rate',float, 'Set the learning rate'),
    ('supervisionRatio', 'Supervision ratio',float, 'Weight of supervision over unsupervised learning'),
    #('theta'
    ]

def test_params(net,p):
    preIsSet = False
    postIsSet = False
    nameIsTaken = False
    nodeList = net.network.getNodes()
    for i in nodeList:
        if i.name == p['preName']:
            preIsSet = True
        elif i.name == p['postName']:
            postIsSet = True
        elif i.name == p['errName']:
            nameIsTaken = True
    if nameIsTaken: return 'The name for error ensemble is already taken'
    if not preIsSet: return 'Must provide the name of an existing pre ensemble'
    if not postIsSet: return 'Must provide the name of an existing post ensemble'
    if p['N_err'] < 1: return 'Number of dimensions must be at least one'
    
import random
from ca.nengo.model.plasticity.impl import STDPTermination, PlasticEnsembleImpl
from ca.nengo.model import StructuralException
import nef
import numeric
from java.util import ArrayList
from java.util import HashMap
def make(net,errName='error', N_err=50, preName='pre', postName='post', rate=5e-4, supervisionRatio=0.5, theta=None, weight=None):

    # get pre and post ensembles from their names
    pre = net.network.getNode(preName)
    post = net.network.getNode(postName)
    
    # Create error ensemble
    try:
        net.get(errName) # check if it already exists
    except StructuralException:
        net.make(errName, N_err, post.dimension)
    
    # modulatory termination (find unused termination)
    count=0
    while 'mod_%02d'%count in [t.name for t in post.terminations]:
        count=count+1
    modname = 'mod_%02d'%count
    if net.get(errName).dimension == post.dimension:
        modweights = numeric.eye(post.dimension)
    else:
        modweights = [[1 for i in range(net.get(errName).dimension)] for j in range(post.dimension)]
    mterm = post.addDecodedTermination(modname, modweights, 0.005, True)
    
    # random weight matrix to initialize projection from pre to post
    if weight == None:
        def rand_weights(w):
            for i in range(len(w)):
                for j in range(len(w[0])):
                    w[i][j] = random.uniform(-1e-3,1e-3)
            return w
        weight = rand_weights(numeric.zeros((post.neurons, pre.neurons)).tolist())
    
    # non-decoded termination (to learn transformation)
    count = 0
    prename = pre.getName()
    while '%s_%02d'%(prename,count) in [t.name for t in post.terminations]:
        count=count+1
    prename = '%s_%02d'%(prename, count)

    lterm = post.addHPESTermination(prename, weight, 0.005, False, theta)
    
    # Add projections
    net.connect(errName, post.getTermination(modname))
    net.connect(pre.getOrigin('AXON'),post.getTermination(prename))

    # Set learning rule on the non-decoded termination
    net.learn(post,prename,modname,rate=rate,supervisionRatio=supervisionRatio)

    if net.network.getMetaData("learnedterm") == None:
        net.network.setMetaData("learnedterm", HashMap())
    learnedterms = net.network.getMetaData("learnedterm")

    learnedterm=HashMap(7)
    learnedterm.put("errName", errName)
    learnedterm.put("N_err", N_err)
    learnedterm.put("preName", preName)
    learnedterm.put("postName", postName)
    learnedterm.put("rate", rate)
    learnedterm.put("theta", theta)
    learnedterm.put("supervisionRatio", supervisionRatio)

    learnedterms.put(errName, learnedterm)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(errName)

    if net.network.getMetaData("templateProjections") == None:
        net.network.setMetaData("templateProjections", HashMap())
    templateproj = net.network.getMetaData("templateProjections")
    templateproj.put(errName, postName)
    templateproj.put(preName, postName)
    
    return lterm,mterm
    
