N=60
D=2

import nef.nef_theano as nef
import random
import math

random.seed(37)

net=nef.Network('Learn Product') #Create the network object

# Create input and output populations.
net.make('pre',N,D)#Make a population with 60 neurons, D dimensions
net.make('post',N,1) #Make a population with 60 neurons, 1 dimensions
net.make('error',N,1) #Make a population with 60 neurons, 1 dimensions

# Create a random function input.
net.make_input('input', math.sin)

net.connect('input','pre')

# Create a modulated connection between the 'pre' and 'post' ensembles.
net.learn(error='error', pre='pre', post='post',
    rate=5e-7) #Make an error population with 100 neurons, and a learning 
               #rate of 5e-7

# Set the modulatory signal to compute the desired function
def product(x):
    product=1.0
    for xx in x: product*=xx
    return product

net.connect('pre', 'error', func=product)
net.connect('post', 'error', weight=-1)

net.run(1) # run for 1 second
