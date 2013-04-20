import nef.nef_theano as nef

net=nef.Network('Multiplication') #Create the network object

net.make_input('input A',[8]) #Create a controllable input function 
                                    #with a starting value of 8
net.make_input('input B',[5]) #Create a controllable input function 
                                    #with a starting value of 5
net.make('A',100,1,radius=10) #Make a population with 100 neurons,
                                           #1 dimensions, a radius of 10
                                           #(default is 1)
net.make('B',100,1,radius=10) #Make a population with 100 neurons, 1 dimensions, a
                      #radius of 10 (default is 1)                    
net.make('Combined',225,2,radius=15) 
                #Make a population with 225 neurons, 2 dimensions, and set a 
                #larger radius (so 10,10 input still fits within the circle 
                #of that radius)
net.make('D',100,1,radius=100) #Make a population with 100 neurons, 1 dimensions, a 
                      #radius of 10 (default is 1)
                      
net.connect('input A','A') #Connect all the relevant objects
net.connect('input B','B')
net.connect('A','Combined', index_post=0) #Connect with the given 1x2D mapping matrix
net.connect('B','Combined', index_post=1)

def product(x):
    return x[0]*x[1]
net.connect('Combined','D',func=product) #Create the output connection mapping the 
                              #1D function 'product'

net.run(1) # run for 1 second
