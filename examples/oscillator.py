import nef.nef_theano as nef

net=nef.Network('Oscillator') #Create the network object

net.make_input('input',[1,0],
    zero_after_time=0.1) #Create a controllable input function with a 
                         #starting value of 1 and zero, then make it go 
                         #to zero after .1s
                         
net.make('A',200,2) #Make a population with 200 neurons, 2 dimensions

net.connect('input','A')
net.connect('A','A',[[1,1],[-1,1]],pstc=0.1) #Recurrently connect the population 
                                         #with the connection matrix for a 
                                         #simple harmonic oscillator mapped 
                                         #to neurons with the NEF

net.run(1) # run for 1 second
