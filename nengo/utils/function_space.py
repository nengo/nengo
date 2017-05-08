from __future__ import absolute_import

import numpy as np

from nengo.utils.numpy import array


def gen_args(n, *arg_dists):
    """
    Helper function to sample arguments given their distributions.

    Parameters:
    -----------
     n: int,
       number of functions to generate arguments for

    arg_dists: instances of nengo distributions
       distributions to sample arguments (eg. mean of a gaussian function) from

    Returns:
    -------
     ndarray of shape (n, number of passed distributions)
    """

    return np.array([arg_dist.sample(n) for arg_dist in arg_dists]).T


def gen_funcs(base_func, args, points):
    """
    Helper function to generate variations of ``base_func`` evaluated on
    points.

    Parameters:
    -----------

    base_func: callable,
       real-valued function to be used as a basis, ex. gaussian

    args: ndarray of arguments to be bound to the function
       distributions to sample arguments (eg. mean of a gaussian function) from
       shape should be (# of functions, # of arguments)

    points:
       the 'domain' of the functions

    Returns:
    --------
    ndarray of shape (len(funcs), len(points)).
    """

    # make a list of callable functions that have their parameters fixed
    # to the arguments given
    funcs = []
    for i in range(args.shape[0]):
        def func(points, i=i):
            f_args = [points]
            f_args.extend(args[i])
            # since function outputs are 1D, flatten
            return base_func(*f_args).flatten()
        funcs.append(func)

    # evaluate the functions on the points given
    dtype = funcs[0](points).dtype
    values = np.empty((len(funcs), len(points)), dtype=dtype)
    for i, func in enumerate(funcs):
        values[i, :] = func(points).flatten()
    return values


def mesh_and_stack(*axes):
    """
    Helper function to mesh axes into a grid and stack them.

    Can be used to generate a mesh of arguments for functions, or a mesh of
    domain points. The stacking is to get it in the right shape to be used
    in this file.

    Returns:
    -------
    ndarray of shape (product of axes' lengths, # of axes).
    """

    grid = np.meshgrid(*axes)
    return np.vstack(map(np.ravel, grid)).T


def uniform_cube(domain_dim, radius=1, d=0.001):
    """Returns uniformly spaced points in a hypercube.

    The hypercube is defined by the given radius and dimension.

    Parameters:
    ----------
    domain_dim: int
       the dimension of the domain

    radius: float, optional
       2 * radius is the length of a side of the hypercube

    d: float, optional
       the discretization spacing (a small float)

    Returns:
    -------
    ndarray of shape ((2 * radius/d) ^ domain_dim), domain_dim)

    """

    if domain_dim == 1:
        domain_points = np.arange(-radius, radius, d)
        domain_points = array(domain_points, min_dims=2)
    else:
        axis = np.arange(-radius, radius, d)
        domain_points = mesh_and_stack(*[axis for _ in range(domain_dim)])
    return domain_points


class Func_Space(object):
    """Base class for using function spaces in nengo.

    Parameters:
    -----------
    encoder_fns: ndarray
      The functions that will serve as encoders
    """

    def __init__(self, encoder_fns):
        self.fns = encoder_fns
        self.n_functions = len(self.fns)
        self.n_points = self.fns.shape[1]

    def get_encoder_fns(self):
        return self.fns

    def get_basis(self):
        raise NotImplementedError("Must be implemented by subclasses")

    def reconstruct(self, coefficients):
        raise NotImplementedError("Must be implemented by subclasses")

    def get_encoders(self):
        raise NotImplementedError("Must be implemented by subclasses")

    def project(self, signal):
        raise NotImplementedError("Must be implemented by subclasses")


class SVD_FS(Func_Space):
    """A function space subclass where the basis is derived from the SVD.

    Takes the functions evaluated on the domain and makes an orthonormal bases
    by using the SVD. Member functions can be used to project or reconstruct
    relative to this basis.

    Parameters:
    -----------

    domain_dim: int,
      The dimension of the domain on which the function space is defined

    d: float, optional
       the discretization factor (used in spacing the domain points)

    n_basis: int, optional
      Number of orthonormal basis functions to use
    """

    def __init__(self, encoder_fns, domain_dim, d, n_basis=20):

        super(SVD_FS, self).__init__(encoder_fns)
        self.n_basis = n_basis
        self.dx = d ** domain_dim  # volume element for integration

        # create orthonormal basis
        self.U, self.S, self.V = np.linalg.svd(self.fns)
        self.basis = self.V[:self.n_basis] / np.sqrt(self.dx)

    def select_top_basis(self, n):
        """Reselects the top n basis functions."""
        self.n_basis = n
        self.basis = self.V[:self.n_basis] / np.sqrt(self.dx)

    def get_basis(self):
        return self.basis

    def singular_values(self):
        return self.S

    def reconstruct(self, coefficients):
        """Linear combination of the basis functions"""
        return np.dot(self.basis.T, coefficients)

    def encoders(self):
        """Project encoder functions onto basis to get encoder vectors."""
        return self.project(self.fns)

    def project(self, signal):
        """Project a given signal onto basis to get signal coefficients.
           Size returned is (n_signals, n_basis)"""
        return np.dot(self.basis, signal.T).T * self.dx


class Fourier(Func_Space):
    """A function space subclass that uses the Fourier basis."""

    def __init__(self, encoder_fns, n_basis=20):

        self.N = encoder_fns.shape[1]
        # create the fourier basis for completeness, not used for any
        # computation
        k = np.arange(self.N)
        basis = []
        for n in range(n_basis):
            basis.append(np.exp(-1j * 2 * np.pi * n * k / self.N))
        self.basis = np.array(basis)
        self.n_basis = n_basis
        super(Fourier, self).__init__(encoder_fns)

    def threshold(self, thresh, signal):
        coeff = self.project(signal)
        return coeff[np.where(np.abs(coeff) > thresh)]

    def get_basis(self):
        return self.basis

    def reconstruct(self, coefficients):
        """inverse fourier transform"""
        return np.fft.irfft(coefficients, self.N)

    def encoders(self):
        """Apply the fourier transform to the encoder functions."""
        return self.project(self.fns)

    def project(self, signal):
        """Apply the Discrete fourier transform to the signal"""
        # throw out higher frequency coefficients
        return np.fft.rfft(signal.T)[:self.n_basis]


def sample_comb(base_func, k, domain, *arg_dists):
    """
    Helper function for making a random linear combination of a function.

    Creates a sample linear combination by summing k random variations
    of the base function used for tiling
    """
    args = gen_args(k, *arg_dists)
    funcs = gen_funcs(base_func, args, domain)
    # evaluate them on the domain and add up
    sample_input = np.sum(funcs, axis=0)
    return sample_input


def sample_eval_points(base_func, FS, n_eval_points, k, domain, *arg_dists):
    """
    Helper function for making eval points as linear combinations from a
    function.

    Create evalutations points as random linear combinations using the
    ``sample_comb`` helper function"""

    funcs = []
    for _ in range(n_eval_points):
        funcs.append(sample_comb(base_func, k, domain, *arg_dists))
    return FS.project(np.array(funcs))
