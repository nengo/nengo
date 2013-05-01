import nengo
from nengo.networks import array
from nengo.helpers import gen_transform

## This example demonstrates how to create a neural model to perform a matrix multiplication
##   on arbitrarily sized matricies.
##
## Network diagram:
##
##      [Input A] ---> {A} --.
##                           v
##                          {C} ---> {D}
##                           ^
##      [Input B] ---> {B} --'
##
##
## Network behaviour:
##   A = Input_A
##   B = Input_B
##   C = [[A0,B0],[A1,B0],[A2,B0], ... , [AN,BM]] (see Notes)
##   D = A * B
##
## Notes: 
##   - To compute the matrix multiplication, the idea is to calculate the all of the necessary
##     multiplications in parallel, and then add up the results later (in population D). This
##     means we need D1*D2*D3 multiplications (in population C) to do this.
##

# Define model parameters
D1 = 1                                                   # Matrix A is D1xD2
D2 = 2                                                   # Matrix B is D2xD3
D3 = 2                                                   # Result   is D1xD3
radius = 1                                               # Values should stay within the range 
                                                         #   (-radius, radius)

# Create the nengo model
model = nengo.Model('Matrix Multiplication') 


# Create the model inputs
model.make_node('Input A', [0]*D1*D2)                    # Create the inputs to the model. Note the
model.make_node('Input B', [0]*D2*D3)                    #   dimensionality of each input (see model
                                                         #   parameters)

# Create neuronal network arrays 
array.make(model, 'A', D1*D2, 50, 1, radius = radius)    # Create the network arrays that represent
array.make(model, 'B', D2*D3, 50, 1, radius = radius)    #   the inputs to the model
C = array.make(model, 'C', D1*D2*D3, 200, 2,             # The C network array holds the intermediate 
               radius = 1.5 * radius,                    #   D1*D2*D3 product calculations needed to
                                                         #   multiply the 2 matrices together
C.encoders = [[1,1],[1,-1],[-1,1],[-1,-1]]               # Set the encoders of ensembles in the C 
                                                         #   network array such that both A and B inputs
                                                         #   are represented equally in C
array.make(model, 'D', D1*D3, 50, 1, radius = radius)    # Create the output population that represents
                                                         #   the resulting matrix multiplication

# Create the connections within the model
array.connect(model, 'Input A', 'A')                     # Connect the inputs to the appropriate network
array.connect(model, 'Input B', 'B')                     #   arrays

transformA = [[0]*(D1*D2) for i in range(D1*D2*D3*2)]    # Determine the transformation matrices to get
transformB = [[0]*(D2*D3) for i in range(D1*D2*D3*2)]    #  the correct pairwise products computed.  This
for i in range(D1):                                      #  looks a bit like black magic but if you 
    for j in range(D2):                                  #  manually try multiplying two matrices together,
        for k in range(D3):                              #  you can see the underlying pattern.  Basically,
            transformA[(j+k*D2+i*D2*D3)*2][j+i*D2] = 1   #  we need to build up D1*D2*D3 pairs of numbers 
            transformB[(j+k*D2+i*D2*D3)*2+1][k+j*D3] = 1 #  in C to compute the product of. If i,j,k are 
                                                         #  the indexes into the D1*D2*D3 products, we want 
                                                         #  to compute the product of element (i,j) in A 
                                                         #  with the element (j,k) in B.  The index in A of 
                                                         #  (i,j) is j+i*D2 and the index in B of (j,k) is 
                                                         #  k+j*D3. The index in C is j+k*D2+i*D2*D3, 
                                                         #  multiplied by 2 since there are two values per 
                                                         #  ensemble.  We add 1 to the B index so it goes into
                                                         #  the second value in the ensemble.  

array.connect(model, 'A', 'C', transform = transformA)   # Connect the A and B networks to the C network
array.connect(model, 'B', 'C', transform = transformB)   #   using the transformation matrices specified
                                                         #   above

def product(x):                                          # Define the product function
    return x[0] * x[1]
array.connect(model, 'C', 'D', transform = gen_transform(index_post = [i/D2 for i in range(D1*D2*D3)]), 
              func = product)                            # Create the output connection that calculates
                                                         #   all of the multiplications needed and maps
                                                         #   the result of the multiplications to the
                                                         #   matrix multiplication result in D
# Build the model
model.build()

# Run the model
model.run(1)                                             # Run the model for 1 second

