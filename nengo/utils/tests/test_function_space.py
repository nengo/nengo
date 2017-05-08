import numpy as np

import nengo

from nengo.utils.function_space import *
from nengo.dists import Uniform


def gaussian(points, center):
    return np.exp(-(points - center)**2 / (2 * 0.2 ** 2))


def test_function_repr(Simulator, nl, plt):

    # parameters and setup
    n_neurons = 2500  # number of neurons
    domain_dim = 1
    arg_dist = Uniform(-1, 1)
    args = gen_args(n_neurons, arg_dist)
    domain = uniform_cube(domain_dim=domain_dim)
    fns_vals = gen_funcs(gaussian, args, domain)

    # function space object
    FS = SVD_FS(fns_vals, d=0.001, domain_dim=domain_dim, n_basis=20)

    # test input is a gaussian bumps function
    # generate a bunch of gaussian functions
    input_func = sample_comb(gaussian, 4, domain, arg_dist)
    input_func /= np.linalg.norm(input_func)

    # evaluation points are gaussian bumps functions
    n_eval_points = 2000
    eval_points = sample_eval_points(gaussian, FS, n_eval_points, 4, domain,
                                     arg_dist)

    # vector space coefficients
    signal_coeffs = FS.project(input_func)
    encoders = FS.encoders()

    f_radius = np.linalg.norm(signal_coeffs)  # radius to use for ensemble

    with nengo.Network() as model:
        # represents the function
        f = nengo.Ensemble(n_neurons=n_neurons, dimensions=FS.n_basis,
                           encoders=encoders, radius=f_radius,
                           eval_points=eval_points, label='f')
        signal = nengo.Node(output=signal_coeffs)
        nengo.Connection(signal, f)

        probe_f = nengo.Probe(f, synapse=0.1)

    sim = Simulator(model)
    sim.run(0.2)

    reconstruction = FS.reconstruct(sim.data[probe_f][100])
    true_f = input_func.flatten()

    plt.saveas = "func_repr.pdf"

    plt.plot(domain, reconstruction, label='model_f')
    plt.plot(domain, true_f, label='true_f')
    plt.legend(loc='best')

    assert np.allclose(true_f, reconstruction, atol=0.2)


# def test_fourier_basis(plt):
#     """Testing fourier basis, not in neurons"""

#     # parameters
#     domain_dim = 1

#     FS = Fourier(domain_dim)

#     # test input is a gaussian bumps function
#     # generate a bunch of gaussian functions
#     gaussians = generate_functions(gaussian, 4, Uniform(-1, 1))
#     # evaluate them on the domain and add up
#     true = np.sum([func(FS.domain) for func in gaussians], axis=0)
#     model = FS.reconstruct(FS.signal_coeffs(true))

#     plt.figure('Testing Fourier Basis')
#     plt.plot(FS.domain, true, label='Function')
#     plt.plot(FS.domain, model, label='reconstruction')
#     plt.legend(loc='best')
#     plt.savefig('utils.test_function_space.test_fourier_basis.pdf')

#     # clip ends because of Gibbs phenomenon
#     assert np.allclose(true[200:-200], model[200:-200], atol=0.2)


def function_gen_plot(plt):
    """Plot the output of function generation to check if it works"""
    arg_dist = Uniform(-1, 1)
    args = gen_args(n_neurons, arg_dist)
    domain = uniform_cube(domain_dim=domain_dim)
    fns_vals = gen_funcs(gaussian, args, domain)
    plt.plot(domain, fns_vals, label='function_values')


def uniform_cube_plot(plt):
    """Plot the output of domain point generation to check if it works"""
    points = uniform_cube(2, 1, 0.1)
    plt.scatter(points[0, :], points[1, :], label='points')
