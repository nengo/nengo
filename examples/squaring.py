import nef.nef_theano as nef

net=nef.Network('Squaring') #Create the network object

net.make_input('input',[0]) #Create a controllable input function 
                            #with a starting value of 0
net.make('A',100,1) #Make a population with 100 neurons, 1 dimension
net.make('B',100,1) #Make a population with 
                                                #100 neurons, 1 dimensions
net.connect('input','A') #Connect the input to A
net.connect('A','B',func=lambda x: x[0]*x[0]) #Connect A and B with the 
                                          #defined function approximated 
                                          #in that connection

net.run(1) # run for 1 second
