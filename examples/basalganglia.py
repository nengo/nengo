import nef.nef_theano as nef
import nps

D=5
net=nef.Network('Basal Ganglia') #Create the network object

net.make_input('input',[0]*D) #Create a controllable input function 
                              #with a starting value of 0 for each of D
                              #dimensions
net.make('output',1,D,mode='direct')  
                 #Make a population with 100 neurons, 5 dimensions, and set 
                 #the simulation mode to direct
nps.basalganglia.make_basal_ganglia(net,'input','output',D,same_neurons=False,
    neurons=50)  #Make a basal ganglia model with 50 neurons per action
net.add_to_nengo()

