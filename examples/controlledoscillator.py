import nef.nef_theano as nef

speed=10  #Base frequency of oscillation
tau=0.1   #Recurrent time constant

net = nef.Network('Controlled Oscillator')

#Make controllable inputs
net.make_input('Start', [1,0], zero_after_time=0.15) #Kick it to get it going
net.make_input('Speed', [1]) #Control the speed interactively

#Make two populations, one for freq control, and one the oscillator
net.make('Oscillator', 500, 3, radius=1.7)
net.make('SpeedNeurons', 100, 1)

#Connect the elements of the network
net.connect('Start', 'Oscillator', index_post=[0,1])
net.connect('Speed', 'SpeedNeurons')
net.connect('SpeedNeurons', 'Oscillator', index_post=[2])

#Define the nonlinear interactions in the state space of the oscillator
def controlled_path(x):
    return x[0]+x[2]*speed*tau*x[1], x[1]-x[2]*speed*tau*x[0], 0
        
net.connect('Oscillator', 'Oscillator', func=controlled_path, pstc=tau)

net.run(1) # run for 1 second
