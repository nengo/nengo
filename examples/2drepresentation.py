import numpy as np
import matplotlib.pyplot as plt

import nengo

"""
This example demonstrates how to create a neuronal ensemble
that represents a two-dimensional signal.

Network diagram:

     [Input] ---> (Neurons)

Network behaviour:
  Neurons = Input

"""

model = nengo.Model('2D Representation')

model.make_node('Sin', output=np.sin)
model.make_node('Cos', output=np.cos)

# Create a population with 100 LIF neurons representing 2 dimensions
e = model.make_ensemble('Neurons', nengo.LIF(100), 2)

# Connect the input to the neuronal population
model.connect('Sin', 'Neurons', transform=[[1], [0]])
model.connect('Cos', 'Neurons', transform=[[0], [1]])

model.probe('Sin')
model.probe('Cos')
model.probe('Neurons')
model.probe(e.decoded_output)

# Run the model for 1 second
model.run(5)

print model.data.keys()

t = model.data['simtime']

plt.plot(t, model.data['Sin'], label="Sine")
plt.plot(t, model.data['Cos'], label="Cosine")
plt.plot(t, model.data['Neurons'], label="Neuron approximation")
plt.plot(t, model.data['Neurons.decoded_output'])

plt.legend()
plt.show()
