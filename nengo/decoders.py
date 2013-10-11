"""
This file contains functions concerned with solving for decoders
or other types of weight matrices.

The idea is that as we develop new methods to find decoders either more
quickly or with different constraints (e.g., L1-norm regularization),
the associated functions will be placed here.
"""

import numpy as np


def sample_hypersphere(dimensions, n_samples, rng, surface=False):
    """Generate sample points from a hypersphere.

    Returns float array of sample points: dimensions x n_samples

    """
    samples = rng.randn(n_samples, dimensions)

    # normalize magnitude of sampled points to be of unit length
    norm = np.sum(samples * samples, axis=1)
    samples /= np.sqrt(norm)[:, np.newaxis]

    if surface:
        return samples

    # generate magnitudes for vectors from uniform distribution
    scale = rng.rand(n_samples, 1) ** (1.0 / dimensions)

    # scale sample points
    samples *= scale

    return samples

# def gen_eval_points(n, d, radius=1.0, method='uniform'):
#     """Various methods for randomly generating evaluation points.

#     TODO: this method is currently not used anywhere!
#     """

#     method = method.lower()
#     if method == 'uniform':
#         ### pick points from a Gaussian distribution, then normalize the
#         ### vector lengths so they are uniformly distributed
#         radii = np.random.uniform(low=0, high=1, size=n)
#         points = np.random.normal(size=(n,d))
#         points *= (radii / np.sqrt((points**2).sum(-1)))[:,None]
#     elif method in ['gaussian', 'normal']:
#         ### pick the points from a Gaussian distribution
#         points = np.random.normal(scale=radius, size=(n,d))
#     else:
#         ### TODO: other methods could allow users to specify different radii
#         ### for different dimensions.
#         raise ValueError("Unrecognized evaluation point generation method")

#     return points





# -- James and Terry arrived at this by eyeballing some graphs.
#    Not clear if this should be a constant at all, it
#    may depend on fn being estimated, number of neurons, etc...
DEFAULT_RCOND = 0.01

# def solve_decoders(activities, targets, method='lstsq'):
#     if method == 'lstsq':
#         weights, res, rank, s = np.linalg.lstsq(activities, targets,
#                                                 rcond=DEFAULT_RCOND)
#     elif method == 'cholesky':
#     elif method == 'eigh':
#         sigma = 0.1 * activities.max()
#         weights = eigh(activites, targets, sigma=sigma)

def timer(label):
    import time
    def deco(f):
        def rval(*args, **kwargs):
            t0 = time.time()
            foo = f(*args, **kwargs)
            t1 = time.time()
            print 'timer "%s" call took: %.2f' % (label, t1 - t0)
            return foo
        return rval
    return deco


@timer('least_squares')
def least_squares(activities, targets, rcond=DEFAULT_RCOND, noise_amp=0.1):
    noise = np.random.randn(*activities.shape) * activities.max() * noise_amp
    activities += noise
    weights, res, rank, s = np.linalg.lstsq(activities, targets, rcond=rcond)
    return weights.T


from scipy.optimize import fmin_l_bfgs_b
@timer('ridge_regression')
def ridge_regression(activities, targets, l2_penalty=1.0, maxfun=100):
    M, N = activities.shape
    M_, D = targets.shape
    assert M == M_
    def func(theta):
        dec = theta.reshape(N, D)
        err = np.dot(activities, dec) - targets
        ml_cost = np.sum(err ** 2)
        l2_cost = np.sum(theta ** 2)
        cost = ml_cost + l2_penalty * l2_cost
        grad = np.dot(activities.T, err)
        return cost, grad.ravel()
    theta_opt, _, _ = fmin_l_bfgs_b(func=func, x0=np.zeros(N * D),
                                    maxfun=maxfun)
    return theta_opt.reshape(N, D).T

@timer('auto_ridge_regression')
def auto_ridge_regression(activities, targets, maxfun=100):
    l2_penalty = 100.0
    costs = [float('inf')]
    decs = [None]
    penalties = [None]
    n_fit = int(.8 * len(activities))
    while l2_penalty >= .01:
        penalties.append(l2_penalty)
        decs.append(
            ridge_regression(
                activities[:n_fit], targets[:n_fit], l2_penalty, maxfun))
        err = np.dot(activities[n_fit:], decs[-1].T) - targets[n_fit:]
        costs.append(((err) ** 2).sum())
        if costs[-1] > costs[-2]:
            break
        l2_penalty /= 10
    print 'auto_ridge_regression: best l2_penalty', penalties[-2]
    return decs[-2]






def regularizationParameter(sigma, Neval):
    return sigma**2 * Neval


def eigh(A, b, sigma):
    """
    Solve the given linear system(s) using the eigendecomposition.
    """

    m,n = A.shape
    reglambda = regularizationParameter(sigma, m)

    transpose = m < n
    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T) + reglambda * np.eye(m)
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A) + reglambda * np.eye(n)
        b = np.dot(A.T, b)

    e,V = np.linalg.eigh(G)
    eInv = 1. / e

    x = np.dot(V.T, b)
    x = eInv[:,None] * x if len(b.shape) > 1 else eInv * x
    x = np.dot(V, x)

    if transpose:
        x = np.dot(A.T, x)

    return x

def cholesky(A, b, sigma):
    """
    Solve the given linear system(s) using the Cholesky decomposition
    """
    # sigma = 0.1 * activities.max()
    # weights = cholesky(activities, targets, sigma=sigma)

    m,n = A.shape
    reglambda = regularizationParameter(sigma, m)

    transpose = m < n
    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T) + reglambda * np.eye(m)
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A) + reglambda * np.eye(n)
        b = np.dot(A.T, b)

    L = np.linalg.cholesky(G)
    L = np.linalg.inv(L.T)

    x = np.dot(L.T, b)
    x = np.dot(L, x)

    if transpose:
        x = np.dot(A.T, x)

    return x


# def _conjgrad_iters(A, b, x, maxiters=None, atol=1e-6, btol=1e-6, rtol=1e-6):
#     """
#     Perform conjugate gradient iterations
#     """

#     if maxiters is None:
#         maxiters = b.shape[0]

#     r = b - A(x)
#     p = r.copy()
#     rsold = np.dot(r,r)
#     normA = 0.0
#     normb = npl.norm(b)

#     for i in range(maxiters):
#         Ap = A(p)
#         alpha = rsold / np.dot(p,Ap)
#         x += alpha*p
#         r -= alpha*Ap

#         rsnew = np.dot(r,r)
#         beta = rsnew/rsold

#         if np.sqrt(rsnew) < rtol:
#             break

#         if beta < 1e-12: # no perceptible change in p
#             break

#         p = r + beta*p
#         rsold = rsnew

#     # print "normA est: %0.3e" % (normA)
#     return x, i+1

# def conjgrad(A, b, sigma, x0=None, maxiters=None, tol=1e-2):
#     """
#     Solve the given linear system using conjugate gradient
#     """

#     m,n = A.shape
#     D = b.shape[1] if len(b.shape) > 1 else 1
#     damp = m*sigma**2

#     G = lambda x: (np.dot(A.T,np.dot(A,x)) + damp*x)
#     b = np.dot(A.T,b)

#     ### conjugate gradient
#     x = np.zeros((n,D))
#     itns = []

#     for i in range(D):
#         xi = getcol(x0, i).copy() if x0 is not None else x[:,i]
#         bi = getcol(b, i)
#         xi, itn = _conjgrad_iters(G, bi, xi, maxiters=maxiters, rtol=tol*np.sqrt(m))
#         x[:,i] = xi
#         itns.append(itn)

#     if D == 1:
#         return x.ravel(), itns[0]
#     else:
#         return x, itns
