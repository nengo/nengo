"""Utilities for optimizing neuron populations and networks."""

from __future__ import absolute_import

import numpy as np
import scipy.optimize
from scipy.special import beta, betainc

import nengo
from nengo.dists import SubvectorLength
from nengo.params import Deferral

# TODO documentation in this file

class RadiusForUnitVector(Deferral):
    """
    Parameters
    ----------
    sp_dimensions : int
        Dimensionality of the complete semantic pointer/unit vector.
    sp_subdimensions : int, optional
        Dimensionality of the subvectors represented by the ensembles.
    """
    def __init__(self, sp_dimensions, sp_subdimensions=1, magnitude=1.):
        super(RadiusForUnitVector, self).__init__()
        self.sp_dimensions = sp_dimensions
        self.sp_subdimensions = sp_subdimensions
        self.magnitude = magnitude

    def default_fn(self, model, ens):
        distortion = get_distortion(model.simulator.__class__, ens)
        return self.magnitude * find_optimal_radius(
            distortion, self.sp_dimensions, self.sp_subdimensions)

    def __mul__(self, value):
        return RadiusForUnitVector(
            self.sp_dimensions, self.sp_subdimensions, value * self.magnitude)

    def __rmul__(self, value):
        return RadiusForUnitVector(
            self.sp_dimensions, self.sp_subdimensions, value * self.magnitude)


# TODO: Fixed seed to enable caching?
def get_distortion(Simulator, ens, conn_kwargs=None, seed=None):
    if conn_kwargs is None:
        conn_kwargs = {}

    with nengo.Network(seed=seed, add_to_container=False) as m:
        ens_copy = ens.copy()
        ens_copy.radius = 1.
        conn = nengo.Connection(
            ens_copy,
            nengo.Ensemble(1, ens.dimensions, neuron_type=nengo.Direct()),
            **conn_kwargs)
    sim = Simulator(m)
    return np.mean(np.square(sim.model.params[conn].solver_info['rmses']))


def find_optimal_radius(distortion, sp_dimensions, sp_subdimensions=1):
    """Determines the optimal radius for ensembles when splitting up a
    semantic pointer (unit vector) into subvectors.

    Requires Scipy.

    Parameters
    ----------
    distortion : float
        Estimate of distortion for a radius of 1 to base optimization on.
    sp_dimensions : int
        Dimensionality of the complete semantic pointer/unit vector.
    sp_subdimensions : int, optional
        Dimensionality of the subvectors represented by the ensembles.

    Returns
    -------
    float
        Optimal radius for the representing ensembles.
    """
    res = scipy.optimize.minimize(
        lambda x: sp_subvector_error(
            x, distortion, sp_dimensions, sp_subdimensions),
        0.1, bounds=[(0., 1.)], method='L-BFGS-B')
    return np.asscalar(res.x)


def sp_subvector_error(radius, distortion, sp_dimensions, sp_subdimensions=1):
    """Estimate of representational error of a subvector of a semantic
    pointer (unit vector).

    Requires Scipy.

    Paramaters
    ----------
    radius : float or ndarray
        Radius of the representing ensemble.
    distortion : float
        Estimate of distortion for a radius of 1 to base optimization on.
    sp_dimensions : int
        Dimensionality of the complete semantic pointer/unit vector.
    sp_subdimensions : int, optional
        Dimensionality of the subvector represented by some ensemble.

    Returns
    -------
    Error estimates for representing a subvector with `subdimensions`
    dimensions of a `dimensions` dimensional unit vector with an ensemble
    initialized with of `radius`.
    """
    dist = SubvectorLength(sp_dimensions, sp_subdimensions)
    in_range = _sp_subvector_error_in_range(radius, distortion,
        sp_subdimensions)
    out_of_range = _sp_subvector_error_out_of_range(
        radius, sp_dimensions, sp_subdimensions)
    return dist.cdf(radius) * in_range + (
        1.0 - dist.cdf(radius)) * out_of_range


def _sp_subvector_error_in_range(radius, distortion, subdimensions):
    return radius * radius * subdimensions * distortion


def _sp_subvector_error_out_of_range(radius, dimensions, subdimensions):
    dist = SubvectorLength(dimensions, subdimensions)
    sq_r = radius * radius

    normalization = 1.0 - dist.cdf(radius)
    b = (dimensions - subdimensions) / 2.0
    aligned_integral = beta(subdimensions / 2.0 + 1.0, b) * (1.0 - betainc(
        subdimensions / 2.0 + 1.0, b, sq_r))
    cross_integral = beta((subdimensions + 1) / 2.0, b) * (1.0 - betainc(
        (subdimensions + 1) / 2.0, b, sq_r))

    numerator = (sq_r * normalization + (
        aligned_integral - 2.0 * radius * cross_integral) / beta(
        subdimensions / 2.0, b))
    with np.errstate(invalid='ignore'):
        return np.where(
            numerator > np.MachAr().eps,
            numerator / normalization, np.zeros_like(normalization))
