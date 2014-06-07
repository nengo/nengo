# # Nengo Example: Matrix multiplication
#
# This example demonstrates how to perform general matrix multiplication
# using Nengo.  The matrix can change during the computation, which makes it
# distinct from doing static matrix multiplication with neural connection
# weights (as done in all neural networks).

import nengo
import numpy as np

N = 100
Amat = np.asarray([[.5, -.5]])
Bmat = np.asarray([[0.58, -1.,], [.7, 0.1]])

model = nengo.Network(label='Matrix Multiplication', seed=123)

# Values should stay within the range (-radius,radius)
radius = 1

with model:
    # Make 2 EnsembleArrays to store the input
    A = nengo.networks.EnsembleArray(N, Amat.size, radius=radius, label="A")
    B = nengo.networks.EnsembleArray(N, Bmat.size, radius=radius, label="B")

    # connect inputs to them so we can set their value
    inputA = nengo.Node(Amat.ravel(), label="input A")
    inputB = nengo.Node(Bmat.ravel(), label="input B")
    nengo.Connection(inputA, A.input)
    nengo.Connection(inputB, B.input)
    A_probe = nengo.Probe(A.output, sample_every=0.01, synapse=0.01)
    B_probe = nengo.Probe(B.output, sample_every=0.01, synapse=0.01)

with model:
    # The C matix is composed of populations that each contain
    # one element of A and one element of B.
    # These elements will be multiplied together in the next step.
    C = nengo.networks.EnsembleArray(N,
                                     n_ensembles=Amat.size * Bmat.shape[1],
                                     ens_dimensions=2,
                                     radius=1.5 * radius,
                                     label="C")

# The appropriate encoders make the multiplication more accurate
for ens in C.ensembles:
    ens.encoders = np.tile(
        [[1, 1], [-1, 1], [1, -1], [-1, -1]],
        (ens.n_neurons // 4, 1))

# Determine the transformation matrices to get the correct pairwise
# products computed.  This looks a bit like black magic but if
# you manually try multiplying two matrices together, you can see
# the underlying pattern.  Basically, we need to build up D1*D2*D3
# pairs of numbers in C to compute the product of.  If i,j,k are the
# indexes into the D1*D2*D3 products, we want to compute the product
# of element (i,j) in A with the element (j,k) in B.  The index in
# A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
# The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
# two values per ensemble.  We add 1 to the B index so it goes into
# the second value in the ensemble.
transformA = np.zeros((C.dimensions, Amat.size))
transformB = np.zeros((C.dimensions, Bmat.size))

for i in range(Amat.shape[0]):
    for j in range(Amat.shape[1]):
        for k in range(Bmat.shape[1]):
            tmp = (j + k * Amat.shape[1] + i * Bmat.size)
            transformA[tmp * 2][j + i * Amat.shape[1]] = 1
            transformB[tmp * 2 + 1][k + j * Bmat.shape[1]] = 1

print("A->C")
print(transformA)
print("B->C")
print(transformB)

with model:
    nengo.Connection(A.output, C.input, transform=transformA)
    nengo.Connection(B.output, C.input, transform=transformB)
    C_probe = nengo.Probe(C.output, sample_every=0.01, synapse=0.01)

with model:
    # Now compute the products and do the appropriate summing
    D = nengo.networks.EnsembleArray(N,
                                     n_ensembles=Amat.shape[0] * Bmat.shape[1],
                                     radius=radius,
                                     label="D")

def product(x):
    return x[0] * x[1]

# The mapping for this transformation is much easier, since we want to
# combine D2 pairs of elements (we sum D2 products together)
transformC = np.zeros((D.dimensions, Bmat.size))
for i in range(Bmat.size):
    transformC[i // Bmat.shape[0]][i] = 1
print("C->D")
print(transformC)

with model:
    prod = C.add_output("product", product)

    nengo.Connection(prod, D.input, transform=transformC)
    D_probe = nengo.Probe(D.output, sample_every=0.01, synapse=0.01)


import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.5067785234899329
gui[model].offset = 52.085570469798654,153.26662856543624
gui[inputA].pos = 0.000, 260.000
gui[inputA].scale = 1.000
gui[inputB].pos = 0.000, 390.000
gui[inputB].scale = 1.000
gui[A].pos = 300.000, 65.000
gui[A].scale = 1.000
gui[A].size = 380.000, 210.000
gui[A.ea_ensembles[0]].pos = 300.000, 0.000
gui[A.ea_ensembles[0]].scale = 1.000
gui[A.ea_ensembles[1]].pos = 300.000, 130.000
gui[A.ea_ensembles[1]].scale = 1.000
gui[A.input].pos = 150.000, 65.000
gui[A.input].scale = 1.000
gui[A.output].pos = 450.000, 65.000
gui[A.output].scale = 1.000
gui[B].pos = 300.000, 455.000
gui[B].scale = 1.000
gui[B].size = 380.000, 470.000
gui[B.ea_ensembles[0]].pos = 300.000, 260.000
gui[B.ea_ensembles[0]].scale = 1.000
gui[B.ea_ensembles[1]].pos = 300.000, 390.000
gui[B.ea_ensembles[1]].scale = 1.000
gui[B.ea_ensembles[2]].pos = 300.000, 520.000
gui[B.ea_ensembles[2]].scale = 1.000
gui[B.ea_ensembles[3]].pos = 300.000, 650.000
gui[B.ea_ensembles[3]].scale = 1.000
gui[B.input].pos = 150.000, 455.000
gui[B.input].scale = 1.000
gui[B.output].pos = 450.000, 455.000
gui[B.output].scale = 1.000
gui[C].pos = 790.000, 325.000
gui[C].scale = 1.000
gui[C].size = 380.000, 470.000
gui[C.ea_ensembles[0]].pos = 790.000, 130.000
gui[C.ea_ensembles[0]].scale = 1.000
gui[C.ea_ensembles[1]].pos = 790.000, 260.000
gui[C.ea_ensembles[1]].scale = 1.000
gui[C.ea_ensembles[2]].pos = 790.000, 390.000
gui[C.ea_ensembles[2]].scale = 1.000
gui[ens].pos = 790.000, 520.000
gui[ens].scale = 1.000
gui[C.input].pos = 640.000, 325.000
gui[C.input].scale = 1.000
gui[C.output].pos = 940.000, 260.000
gui[C.output].scale = 1.000
gui[C.product].pos = 940.000, 390.000
gui[C.product].scale = 1.000
gui[D].pos = 1280.000, 325.000
gui[D].scale = 1.000
gui[D].size = 380.000, 210.000
gui[D.ea_ensembles[0]].pos = 1280.000, 260.000
gui[D.ea_ensembles[0]].scale = 1.000
gui[D.ea_ensembles[1]].pos = 1280.000, 390.000
gui[D.ea_ensembles[1]].scale = 1.000
gui[D.input].pos = 1130.000, 325.000
gui[D.input].scale = 1.000
gui[D.output].pos = 1430.000, 325.000
gui[D.output].scale = 1.000
