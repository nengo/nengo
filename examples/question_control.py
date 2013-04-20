D=16  # number of dimensions for representations 
N=100 # number of neurons per dimension

import nef.nef_theano as nef
import nps
import nef.convolution
import hrr
import math
import random

# semantic pointers do not work well with small numbers of dimensions.  
# To keep this example model small enough to be easily run, we have lowered 
# the number of dimensions (D) to 16 and chosen a random number seed for
# which this model works well.
seed=17

net=nef.Network('Question Answering with Control',seed=seed)

# Make a simple node to generate interesting input for the network
random.seed(seed)
vocab=hrr.Vocabulary(D,max_similarity=0.1)
class Input(nef.SimpleNode): 
    def __init__(self,name):
        self.zero=[0]*D
        nef.SimpleNode.__init__(self,name)
        self.RED_CIRCLE=vocab.parse('STATEMENT+RED*CIRCLE').v
        self.BLUE_SQUARE=vocab.parse('STATEMENT+BLUE*SQUARE').v
        self.RED=vocab.parse('QUESTION+RED').v
        self.SQUARE=vocab.parse('QUESTION+SQUARE').v
    def origin_x(self):
        if 0.1<self.t<0.3:
            return self.RED_CIRCLE
        elif 0.35<self.t<0.5:
            return self.BLUE_SQUARE
        elif self.t>0.5 and 0.2<(self.t-0.5)%0.6<0.4:
            return self.RED
        elif self.t>0.5 and 0.4<(self.t-0.5)%0.6<0.6:
            return self.SQUARE
        else:    
            return self.zero

# Add the input to the network            
inv=Input('inv')
net.add(inv)


prods=nps.ProductionSet() #This is an older way of implementing an SPA 
                          #(see SPA routing examples), using the nps 
                          #code directly
prods.add(dict(visual='STATEMENT'),dict(visual_to_wm=True))
prods.add(dict(visual='QUESTION'),dict(wm_deconv_visual_to_motor=True))


subdim=4
model=nps.NPS(net,prods,D,direct_convolution=False,direct_buffer=['visual'],
    neurons_buffer=N/subdim,subdimensions=subdim)
model.add_buffer_feedback(wm=1,pstc=0.4)

net.connect(inv.getOrigin('x'),'buffer_visual')
      
# Rename objects for display purposes
net.network.getNode('prod').name='thalamus'
net.network.getNode('buffer_visual').name='visual'
net.network.getNode('buffer_wm').name='memory'
net.network.getNode('buffer_motor').name='motor'
net.network.getNode('channel_visual_to_wm').name='channel'
net.network.getNode('wm_deconv_visual_to_motor').name='*'
net.network.getNode('gate_visual_wm').name='gate1'
net.network.getNode('gate_wm_visual_motor').name='gate2'

net.add_to_nengo()

