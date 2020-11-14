"""Configuration presets for common use cases."""

from nengo.config import Config
from nengo.dists import Choice, Exponential, Uniform
from nengo.ensemble import Ensemble


def ThresholdingEnsembles(threshold, intercept_width=0.15, radius=1.0):
    """Configuration preset for a thresholding ensemble.

    This preset adjust ensemble parameters for thresholding. The ensemble's
    neurons will only fire for values above threshold. One can either decode
    the represented value (if it is above the threshold) or decode
    a step function if binary classification is desired.

    This preset:

    - Sets intercepts to be between ``threshold`` and ``radius`` with an
      exponential distribution (shape parameter of ``intercept_width``).
      This clusters intercepts near the threshold for better approximation.
    - Sets encoders to 1.
    - Sets evaluation points to be uniformly distributed between
      ``threshold`` and ``radius``.
    - Sets the radius.

    Parameters
    ----------
    threshold : float
        Point at which ensembles should start firing.
    intercept_width : float, optional
        Controls how widely distributed the intercepts are. Smaller values
        give more clustering at the threshold, larger values give a more
        uniform distribution.
    radius : float, optional
        Ensemble radius.

    Returns
    -------
    `nengo.Config`
        Configuration with presets.
    """
    config = Config(Ensemble)
    config[Ensemble].radius = radius
    config[Ensemble].intercepts = Exponential(intercept_width, threshold, radius)
    config[Ensemble].encoders = Choice([[1]])
    config[Ensemble].eval_points = Uniform(threshold / radius, 1)
    return config
