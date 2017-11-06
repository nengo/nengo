"""Configuration presets for common use cases."""

import nengo


def ThresholdingEnsembles(threshold, intercept_width=0.15, radius=1.):
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
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].radius = radius
    config[nengo.Ensemble].intercepts = nengo.dists.Exponential(
        intercept_width, threshold, radius)
    config[nengo.Ensemble].encoders = nengo.dists.Choice([[1]])
    config[nengo.Ensemble].eval_points = nengo.dists.Uniform(
        threshold / radius, 1)
    return config
