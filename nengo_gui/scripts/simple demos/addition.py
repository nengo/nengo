# #Nengo Example: Addition

# In this example, we will construct a network that adds two inputs. The network utilizes two communication channels into the same neural population. Addition is thus somewhat free, since the incoming currents from different synaptic connections interact linearly (though two inputs dont have to combine in this way: see the combining demo).

import nengo
model = nengo.Network(label='Addition')
with model:
    # Create 3 ensembles each containing 100 leaky integrate-and-fire neurons
    A = nengo.Ensemble(100, dimensions=1, label = "A")
    B = nengo.Ensemble(100, dimensions=1, label = "B")
    C = nengo.Ensemble(100, dimensions=1, label = "C")


    # Create input nodes representing constant values
    input_a = nengo.Node(output=0.5, label = "input a")
    input_b = nengo.Node(output=0.3, label = "input b")
    
    # Connect the input nodes to the appropriate ensembles
    nengo.Connection(input_a, A)
    nengo.Connection(input_b, B)
    
    # Connect input ensembles A and B to output ensemble C
    nengo.Connection(A, C)
    nengo.Connection(B, C)


    input_a_probe = nengo.Probe(input_a)
    input_b_probe = nengo.Probe(input_b)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)
    C_probe = nengo.Probe(C, synapse=0.01)
