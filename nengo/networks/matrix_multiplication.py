import numpy as np

import nengo


class MatrixMult(nengo.Network):

    def __init__(self, n_neurons, shape_a, shape_b,
                 label=None, **ens_kwargs):

        if shape_a[1] != shape_b[0]:
            raise ValueError("Matrix dimensions %s and  %s are incompatible"
                             % (shape_a, shape_b))

        size_a = shape_a[0] * shape_a[1]
        size_b = shape_b[0] * shape_b[1]

        # Make 2 EnsembleArrays to store the input
        self.A = nengo.networks.EnsembleArray(n_neurons, size_a, **ens_kwargs)
        self.B = nengo.networks.EnsembleArray(n_neurons, size_b, **ens_kwargs)

        self.input_a = self.A.input
        self.input_b = self.B.input

        # The C matix is composed of populations that each contain
        # one element of A and one element of B.
        # These elements will be multiplied together in the next step.
        self.C = nengo.networks.EnsembleArray(n_neurons, size_a * shape_b[1],
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
        transform_a = np.zeros((self.C.dimensions, size_a))
        transform_b = np.zeros((self.C.dimensions, size_b))

        for i in range(shape_a[0]):
            for j in range(shape_a[1]):
                for k in range(shape_b[1]):
                    tmp = (j + k * shape_a[1] + i * size_b)
                    transform_a[tmp * 2][j + i * shape_a[1]] = 1
                    transform_b[tmp * 2 + 1][k + j * shape_b[1]] = 1

        nengo.Connection(self.A.output, self.C.input, transform=transform_a)
        nengo.Connection(self.B.output, self.C.input, transform=transform_b)

        # Now compute the products and do the appropriate summing
        self.D = nengo.networks.EnsembleArray(n_neurons,
                                              shape_a[0] * shape_b[1],
                                              **ens_kwargs)

        for ens in self.D.ensembles:
            ens.radius = shape_b[0] * ens.radius

        def product(x):
            return x[0] * x[1]

        # The mapping for this transformation is much easier, since we want to
        # combine D2 pairs of elements (we sum D2 products together)
        transform_c = np.zeros((self.D.n_ensembles, self.C.n_ensembles))

        for i in range(size_a * shape_b[1]):
            transform_c[i // shape_b[0]][i] = 1

        prod = self.C.add_output('product', product)
        nengo.Connection(prod, self.D.input, transform=transform_c)

        self.output = self.D.output
