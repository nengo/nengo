N=60
D=1

import nef.nef_theano as nef
import random
import math

random.seed(27)

net=nef.Network('Learn Communication') #Create the network object

# Create input and output populations.
net.make('pre',N,D) #Make a population with 60 neurons, 1 dimensions
net.make('post',N,D) #Make a population with 60 neurons, 1 dimensions
net.make('error',N,D) 

# Create a random function input.
net.make_input('input', math.sin)
               #Create a white noise input function .1 base freq, max 
               #freq 10 rad/s, and RMS of .5; 12 is a seed

net.connect('input','pre')

# Create a modulated connection between the 'pre' and 'post' ensembles.
net.learn(error='error', pre='pre', post='post',
    rate=5e-7) #Make an error population with 100 neurons, and a learning 
               #rate of 5e-7

# Set the modulatory signal.
net.connect('pre', 'error')
net.connect('post', 'error', weight=-1)

net.make('actual error', N, 1) 
net.connect('pre','actual error')
net.connect('post','actual error',weight=-1)

net.run(1) # run for 1 second
