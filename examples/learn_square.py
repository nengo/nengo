N=60
D=2

import nef.nef_theano as nef
import random
import math

random.seed(27)

net=nef.Network('Learn Square') #Create the network object

# Create input and output populations.
net.make('pre',N,D) #Make a population with 60 neurons, 1 dimensions
net.make('post',N,D) #Make a population with 60 neurons, 1 dimensions
net.make('error',N,D) #Make a population with 60 neurons, 1 dimensions

# Create a random function input.
net.make_input('input', math.sin)
               #Create a white noise input function .1 base freq, max 
               #freq 10 rad/s, and RMS of .4; 0 is a seed  

net.connect('input','pre')

# Create a modulated connection between the 'pre' and 'post' ensembles.
net.learn(error='error', pre='pre', post='post',
    rate=5e-7) #Make an error population with 100 neurons, and a learning 
            #rate of 5e-7

# Set the modulatory signal to compute the desired function
def square(x):
    return [xx*xx for xx in x]

net.connect('pre', 'error', func=square)
net.connect('post', 'error', weight=-1)

net.run(1) # run for 1 second
