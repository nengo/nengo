import numpy as np

import nengo

from nengo.networks.function_space import FS_Ensemble
from nengo.utils.function_space import *
from nengo.utils.tests.test_function_space import gaussian
from nengo.dists import Uniform


def test_FS_ensemble(Simulator, nl, plt):

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

    signal_coeffs = FS.project(input_func)
    f_radius = np.linalg.norm(signal_coeffs)  # radius to use for ensemble

    net = FS_Ensemble(FS, eval_points=eval_points, radius=f_radius)

    with net:
        input = nengo.Node(output=input_func)
        nengo.Connection(input, net.input)
        func_probe = nengo.Probe(net.output, synapse=0.1)

    sim = Simulator(net)
    sim.run(0.2)

    reconstruction = sim.data[func_probe][100]
    true_f = input_func

    plt.saveas = "func_repr.pdf"

    plt.plot(domain, reconstruction, label='model_f')
    plt.plot(domain, true_f, label='true_f')
    plt.legend(loc='best')

    assert np.allclose(true_f, reconstruction, atol=0.2)
