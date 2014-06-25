import numpy as np

import nengo


class MatrixMult(nengo.Network):

    def __init__(self, n_neurons, shapeA, shapeB,
                 label=None, **ens_kwargs):

        if shapeA[1] != shapeB[0]:
            raise ValueError("Matrix dimensions %s and  %s are incompatible"
                             % (shapeA, shapeB))

        sizeA = shapeA[0] * shapeA[1]
        sizeB = shapeB[0] * shapeB[1]

        # Make 2 EnsembleArrays to store the input
        self.A = nengo.networks.EnsembleArray(n_neurons, sizeA, **ens_kwargs)
        self.B = nengo.networks.EnsembleArray(n_neurons, sizeB, **ens_kwargs)

        self.inputA = self.A.input
        self.inputB = self.B.input

        # The C matix is composed of populations that each contain
        # one element of A and one element of B.
        # These elements will be multiplied together in the next step.
        self.C = nengo.networks.EnsembleArray(n_neurons, sizeA * shapeB[1],
                                              ens_dimensions=2, **ens_kwargs)

        # The appropriate encoders make the multiplication more accurate
        # set radius to sqrt(radius^2 + radius^2)
        for ens in self.C.ensembles:
            ens.encoders = np.tile([[1, 1], [-1, 1], [1, -1], [-1, -1]],
                                   (ens.n_neurons // 4, 1))
            ens.radius = np.sqrt(2) * ens.radius

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
        transformA = np.zeros((self.C.dimensions, sizeA))
        transformB = np.zeros((self.C.dimensions, sizeB))

        for i in range(shapeA[0]):
            for j in range(shapeA[1]):
                for k in range(shapeB[1]):
                    tmp = (j + k * shapeA[1] + i * sizeB)
                    transformA[tmp * 2][j + i * shapeA[1]] = 1
                    transformB[tmp * 2 + 1][k + j * shapeB[1]] = 1

        nengo.Connection(self.A.output, self.C.input, transform=transformA)
        nengo.Connection(self.B.output, self.C.input, transform=transformB)

        # Now compute the products and do the appropriate summing
        self.D = nengo.networks.EnsembleArray(n_neurons,
                                              shapeA[0] * shapeB[1],
                                              **ens_kwargs)

        for ens in self.D.ensembles:
            ens.radius = shapeB[0] * ens.radius

        def product(x):
            return x[0] * x[1]

        # The mapping for this transformation is much easier, since we want to
        # combine D2 pairs of elements (we sum D2 products together)
        transformC = np.zeros((self.D.n_ensembles, self.C.n_ensembles))

        for i in range(sizeA * shapeB[1]):
            transformC[i // shapeB[0]][i] = 1

        prod = self.C.add_output('product', product)
        nengo.Connection(prod, self.D.input, transform=transformC)

        self.output = self.D.output
