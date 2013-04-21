from .. import nengo as nengo
from ..nengo.networks import array

# Perform matrix multiplication on arbitrary matrices

# Create the nengo model
model = nengo.Model('Matrix Multiplication') 

# Adjust these values to change the matrix dimensions
#  Matrix A is D1xD2
#  Matrix B is D2xD3
#  Result is D1xD3
D1=1
D2=2
D3=2

# Values should stay within the range (-radius, radius)
radius=1

# Create the model inputs
model.make_node('Input A', [0]*D1*D2)
model.make_node('Input B', [0]*D2*D3)

# Create neuronal network arrays 
array.make(model, 'A', D1*D2, 50, 1, radius=radius)
array.make(model, 'B', D2*D3, 50, 1, radius=radius)

# the C matrix holds the intermediate product calculations
#  need to compute D1*D2*D3 products to multiply 2 matrices together
array.make(model, 'C', D1*D2*D3, 200, 2, radius = 1.5 * radius,
           encoders = [[1,1],[1,-1],[-1,1],[-1,-1]])
array.make(model, 'D', D1*D3, 50, 1, radius = radius)

# Create the connections within the model

model.connect('Input A', 'A') # Connect the inputs to the neural network arrays
model.connect('Input B', 'B')

#  determine the transformation matrices to get the correct pairwise
#  products computed.  This looks a bit like black magic but if
#  you manually try multiplying two matrices together, you can see
#  the underlying pattern.  Basically, we need to build up D1*D2*D3
#  pairs of numbers in C to compute the product of.  If i,j,k are the
#  indexes into the D1*D2*D3 products, we want to compute the product
#  of element (i,j) in A with the element (j,k) in B.  The index in
#  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
#  The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
#  two values per ensemble.  We add 1 to the B index so it goes into
#  the second value in the ensemble.  
transformA=[[0]*(D1*D2) for i in range(D1*D2*D3*2)]
transformB=[[0]*(D2*D3) for i in range(D1*D2*D3*2)]
for i in range(D1):
    for j in range(D2):
        for k in range(D3):
            transformA[(j+k*D2+i*D2*D3)*2][j+i*D2]=1
            transformB[(j+k*D2+i*D2*D3)*2+1][k+j*D3]=1

model.connect('A', 'C', transform=transformA) 
model.connect('B', 'C', transform=transformB)

def product(x):
    return x[0]*x[1]
model.connect('C','D', index_post=[i/D2 for i in range(D1*D2*D3)], func=product)

# Build the model
model.build()

# Run the model
model.run(1)                            # Run the model for 1 second

