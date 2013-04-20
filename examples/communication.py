import nef.nef_theano as nef

net=nef.Network('Communications Channel') #Create the network object

net.make_input('input',[0.5]) #Create a controllable input function 
                              #with a starting value of 0.5
                              
net.make('A',100,1)   #Make a population with 100 neurons, 1 dimension
net.make('B',100,1)   #Make a population with 100 neurons, 1 dimension

net.connect('input','A') #Connect all the relevant objects
net.connect('A','B')

net.run(1) # run for 1 second
