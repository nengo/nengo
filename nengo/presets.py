"""Configuration presets for common use cases."""

import nengo


def ThresholdingEnsembles(threshold):
    """Configuration preset for a thresholding ensemble.

    This preset adjust ensemble parameters for thresholding. The ensemble's
    neurons will only fire for values above threshold. One can either decode
    the represented value (if it is above the threshold) or decode
    a step function if binary classification is desired.

    This preset:

    - Sets intercepts to be between ``threshold`` and 1 with an exponential
      distribution (shape parameter of 0.15). This clusters intercepts near
      the threshold for better approximation.
    - Sets encoders to 1.
    - Sets dimensions to 1.
    - Sets evaluation points to be uniformly distributed between
      ``threshold`` and 1.

    Parameters
    ----------
    threshold : float
        Point at which ensembles should start firing.

    Returns
    -------
    `nengo.Config`
        Configuration with presets.
    """
    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].dimensions = 1
    config[nengo.Ensemble].intercepts = nengo.dists.Exponential(
        0.15, threshold, 1.)
    config[nengo.Ensemble].encoders = nengo.dists.Choice([[1]])
    config[nengo.Ensemble].eval_points = nengo.dists.Uniform(threshold, 1.)
    return config
