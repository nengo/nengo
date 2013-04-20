import nef.nef_theano as nef

#This implements the controlled integrator described in the
#book "How to build a brain"
net=nef.Network('Controlled Integrator 2') #Create the network object
net.make_input('input',{0.2:5, 0.3:0, 0.44:-10,
                            0.54:0, 0.8:5, 0.9:0} )  #Create a controllable input 
                                                     #function with a default function
                                                     #that goes to 5 at time 0.2s, to 0 
                                                     #at time 0.3s and so on
                                                     
net.make_input('control',[0])  #Create a controllable input function
                               #with a starting value of 0
net.make('A',225,2,radius=1.5) #Make a population with 225 neurons, 2 dimensions, and a 
                               #larger radius to accommodate large simulataneous inputs
                               
net.connect('input','A',transform=[[0.1],[0]],pstc=0.1) #Connect all the relevant
                                                #objects with the relevant 1x2
                                                #mappings, postsynaptic time
                                                #constant is 10ms
net.connect('control','A',transform=[[0],[1]],pstc=0.1)
def feedback(x):
    return x[0]*x[1]+x[0] #Different than the other controlled integrator
net.connect('A','A',transform=[[1],[0]],func=feedback,pstc=0.1) #Create the recurrent
                                                        #connection mapping the
                                                        #1D function 'feedback'
                                                        #into the 2D population
                                                        #using the 1x2 transform

net.run(1) # run for 1 second
