D=16
subdim=4
N=100
seed=7

import nef.nef_theano as nef
import nef.convolution
import hrr
import math
import random

random.seed(seed)

vocab=hrr.Vocabulary(D,max_similarity=0.1)

net=nef.Network('Question Answering') #Create the network object

net.make('A',1,D,mode='direct') #Make some pseudo populations (so they 
                                  #run well on less powerful machines): 
                                  #1 neuron, 16 dimensions, direct mode
net.make('B',1,D,mode='direct')
net.make_array('C',N,D/subdim,dimensions=subdim,quick=True,radius=1.0/math.sqrt(D)) 
                           #Make a real population, with 100 neurons per 
                           #array element and D/subdim elements in the array
                           #each with subdim dimensions, set the radius as
                           #appropriate for multiplying things of this 
                           #dimension
net.make('E',1,D,mode='direct')
net.make('F',1,D,mode='direct')

conv1=nef.convolution.make_convolution(net,'*','A','B','C',N,
    quick=True) #Make a convolution network using the construct populations
conv2=nef.convolution.make_convolution(net,'/','C','E','F',N,
    invert_second=True,quick=True) #Make a 'correlation' network (by using
                                   #convolution, but inverting the second 
                                   #input)

CIRCLE=vocab.parse('CIRCLE').v #Add elements to the vocabulary to use
BLUE=vocab.parse('BLUE').v
RED=vocab.parse('RED').v
SQUARE=vocab.parse('SQUARE').v
ZERO=[0]*D

# Create the inputs
inputA={}
inputA[0.0]=RED
inputA[0.5]=BLUE
inputA[1.0]=RED
inputA[1.5]=BLUE
inputA[2.0]=RED
inputA[2.5]=BLUE
inputA[3.0]=RED
inputA[3.5]=BLUE
inputA[4.0]=RED
inputA[4.5]=BLUE
net.make_input('inputA',inputA)
net.connect('inputA','A')

inputB={}
inputB[0.0]=CIRCLE
inputB[0.5]=SQUARE
inputB[1.0]=CIRCLE
inputB[1.5]=SQUARE
inputB[2.0]=CIRCLE
inputB[2.5]=SQUARE
inputB[3.0]=CIRCLE
inputB[3.5]=SQUARE
inputB[4.0]=CIRCLE
inputB[4.5]=SQUARE
net.make_input('inputB',inputB)
net.connect('inputB','B')


inputE={}
inputE[0.0]=ZERO
inputE[0.2]=CIRCLE
inputE[0.35]=RED
inputE[0.5]=ZERO
inputE[0.7]=SQUARE
inputE[0.85]=BLUE
inputE[1.0]=ZERO
inputE[1.2]=CIRCLE
inputE[1.35]=RED
inputE[1.5]=ZERO
inputE[1.7]=SQUARE
inputE[1.85]=BLUE
inputE[2.0]=ZERO
inputE[2.2]=CIRCLE
inputE[2.35]=RED
inputE[2.5]=ZERO
inputE[2.7]=SQUARE
inputE[2.85]=BLUE
inputE[3.0]=ZERO
inputE[3.2]=CIRCLE
inputE[3.35]=RED
inputE[3.5]=ZERO
inputE[3.7]=SQUARE
inputE[3.85]=BLUE
inputE[4.0]=ZERO
inputE[4.2]=CIRCLE
inputE[4.35]=RED
inputE[4.5]=ZERO
inputE[4.7]=SQUARE
inputE[4.85]=BLUE

net.make_input('inputE',inputE)
net.connect('inputE','E')


net.add_to_nengo()

