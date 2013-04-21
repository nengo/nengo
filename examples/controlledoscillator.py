from .. import nengo as nengo
import nengo.nef as nef
from nef import * 

speed = 10  # Base frequency of oscillation
tau = 0.1   # Recurrent time constant

model = nef.model.Network('Controlled Oscillator')

# Make controllable inputs
def start_input(t):
    if t < 0.15:
        return [1,0]
    else:
        return [0,0]
model.make_node('Start', start_input) # Kick it to get it going
model.make_node('Speed', [1]) # Control the speed interactively

# Make two populations, one for freq control, and one the oscillator
model.make_ensemble('Oscillator', 500, 3, radius = 1.7)
model.make_ensemble('SpeedNeurons', 100, 1)

# Connect the elements of the network
model.connect('Start', 'Oscillator', transform = nef.gen_transform(index_post = [0,1]))
model.connect('Speed', 'SpeedNeurons')
model.connect('SpeedNeurons', 'Oscillator', transform = nef.gen_transform(index_post = [2]))

# Define the nonlinear interactions in the state space of the oscillator
def controlled_path(x):
    return x[0] + x[2] * speed * tau * x[1], x[1] - x[2] * speed * tau * x[0], 0
        
model.connect('Oscillator', 'Oscillator', func = controlled_path, 
              filter = nef.ExponentialPSC(pstc = tau))

model.run(1) #  run for 1 second
