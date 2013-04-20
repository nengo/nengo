import nef.nef_theano as nef

net=nef.Network('Addition') #Create the network object

net.make_input('input A',[0])  #Create a controllable input function 
                               #with a starting value of 0
net.make_input('input B',[0])  #Create another controllable input 
                               #function with a starting value of 0
                               
net.make('A',100,1) #Make a population with 100 neurons, 1 dimension
net.make('B',100,1)  #Make a population with 100 neurons, 1 dimension
net.make('C',100,1) #Make a population with 100 neurons, 1 dimension

net.connect('input A','A') #Connect all the relevant objects
net.connect('input B','B')
net.connect('A','C')
net.connect('B','C')

net.run(1) # run for 1 second
