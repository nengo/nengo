title='BCM Termination'
label='BCM\nTermination'
icon='learn.png'

description="""<html>This template creates a new connection modulated by unsupervised learning. The presynaptic side is 'pre' and the postsynaptic side is 'post'.</html>"""

params=[
    ('preName','Name of pre ensemble',str, 'Select the name of the pre ensemble (this ensemble must already exist)'),
    ('postName','Name of post ensemble',str,'Select the name of the post ensemble (this ensemble must already exist)'),
    ('rate','Learning rate',float, 'Set the learning rate')
    #('theta','Average postsynaptic activity',)
    ]

def test_params(net,p):
    preIsSet = False
    postIsSet = False
    for i in nodeList:
        if i.name == p['preName']:
            preIsSet = True
        elif i.name == p['postName']:
            postIsSet = True
    if not preIsSet: return 'Must provide the name of an existing pre ensemble'
    if not postIsSet: return 'Must provide the name of an existing post ensemble'

import random
import nef
import numeric
from java.util import ArrayList
from java.util import HashMap
from ca.nengo.util import MU

def make(net, preName='pre', postName='post', rate=5e-4):

    # get pre and post ensembles from their names
    pre = net.network.getNode(preName)
    post = net.network.getNode(postName)

    dim_pre = pre.getDimension()
    dim_post = post.getDimension()

    t = [[0] * dim_pre for i in range(dim_post)]
    index_pre = range(dim_pre)
    index_post = range(dim_post)
    for i in range(max(len(index_pre),len(index_post))):
        ipre = index_pre[i % len(index_pre)]
        ipost = index_post[i % len(index_post)]
        t[ipost][ipre] = 1

    decoder = pre.getOrigin('X').getDecoders()
    encoder = post.getEncoders()
    encoder = MU.prod(encoder, 1.0 / post.getRadii()[0])
    weight = MU.prod(encoder, MU.prod(t, MU.transpose(decoder)))

    # random weight matrix to initialize projection from pre to post
    # def rand_weights(w):
    #     for i in range(len(w)):
    #         for j in range(len(w[0])):
    #             w[i][j] = random.uniform(-1e-3,1e-3)
    #     return w
    # weight = rand_weights(numeric.zeros((post.neurons, pre.neurons)).tolist())

    # non-decoded termination (to learn transformation)
    count = 0
    prename = pre.getName()
    while '%s_%02d' % (prename, count) in [t.name for t in post.terminations]:
        count = count + 1
    prename = '%s_%02d' % (prename, count)

    post.addBCMTermination(prename, weight, 0.005, False, None)
    
    # Add projections
    net.connect(pre.getOrigin('AXON'),post.getTermination(prename))
    
    # Set learning rule on the non-decoded termination
    net.learn(post,prename,None,rate=rate)

    if net.network.getMetaData("bcmterm") == None:
        net.network.setMetaData("bcmterm", HashMap())
    bcmterms = net.network.getMetaData("bcmterm")

    bcmterm = HashMap(4)
    bcmterm.put("preName", preName)
    bcmterm.put("postName", postName)
    bcmterm.put("rate", rate)

    bcmterms.put(prename, bcmterm)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(prename)

    if net.network.getMetaData("templateProjections") == None:
        net.network.setMetaData("templateProjections", HashMap())
    templateproj = net.network.getMetaData("templateProjections")
    templateproj.put(preName, postName)
    
