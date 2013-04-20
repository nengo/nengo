import nef.nef_theano as nef

net=nef.Network('Combining')  #Create the network object

net.make_input('input A',[0]) #Create a controllable input function 
                              #with a starting value of 0
net.make_input('input B',[0]) #Create another controllable input 
                              #function with a starting value of 0
                              
net.make('A',100,1)  #Make a population with 100 neurons, 1 dimension
net.make('B',100,1)  #Make a population with 100 neurons, 1 dimension 
net.make('C',100,2,
    radius=1.5) #Make a population with 100 neurons, 2 dimensions, and set a
                #larger radius (so 1,1 input still fits within the circle of
                #that radius)
net.connect('input A','A') #Connect all the relevant objects (default connection 
                           #is identity)
net.connect('input B','B')
net.connect('A','C', index_post=0) #Connect with the given 1x2D mapping matrix
net.connect('B','C', index_post=1)

net.run(1) # run for 1 second
