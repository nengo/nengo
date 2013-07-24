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

# Create a controllable 2-D input with a starting value of (0,0)
model.make_node('Input', output=[0, 0])

# Create a population with 100 neurons representing 2 dimensions
model.make_ensemble('Neurons', 100, 2)

# Connect the input to the neuronal population
model.connect('Input', 'Neurons')

# Run the model for 1 second
model.run(1)
