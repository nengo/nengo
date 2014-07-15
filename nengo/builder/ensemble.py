import collections
import warnings

import numpy as np

from nengo.builder.builder import Builder
from nengo.builder.operator import Copy, DotInc, Reset
from nengo.builder.signal import Signal
from nengo.ensemble import Ensemble
from nengo.neurons import Direct
from nengo.utils.builder import default_n_eval_points
from nengo.utils.distributions import Distribution
import nengo.utils.numpy as npext


BuiltEnsemble = collections.namedtuple(
    'BuiltEnsemble', ['eval_points', 'encoders', 'intercepts', 'max_rates',
                      'scaled_encoders', 'gain', 'bias'])


def sample(dist, n_samples, rng):
    if isinstance(dist, Distribution):
        return dist.sample(n_samples, rng=rng)
    return np.array(dist)


@Builder.register(Ensemble)  # noqa: C901
def build_ensemble(model, ens):
    """Builds ensemble."""
    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    # Generate eval points
    if isinstance(ens.eval_points, Distribution):
        n_points = ens.n_eval_points
        if n_points is None:
            n_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
        eval_points = ens.eval_points.sample(n_points, ens.dimensions, rng)
        # eval_points should be in the ensemble's representational range
        eval_points *= ens.radius
    else:
        if (ens.n_eval_points is not None
                and ens.eval_points.shape[0] != ens.n_eval_points):
            warnings.warn("Number of eval_points doesn't match "
                          "n_eval_points. Ignoring n_eval_points.")
        eval_points = np.array(ens.eval_points, dtype=np.float64)

    # Set up signal
    model.sig[ens]['in'] = Signal(np.zeros(ens.dimensions),
                                  name="%s.signal" % ens.label)
    model.add_op(Reset(model.sig[ens]['in']))

    # Set up encoders
    if isinstance(ens.neuron_type, Direct):
        encoders = np.identity(ens.dimensions)
    elif isinstance(ens.encoders, Distribution):
        encoders = ens.encoders.sample(ens.n_neurons, ens.dimensions, rng=rng)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Determine max_rates and intercepts
    max_rates = sample(ens.max_rates, ens.n_neurons, rng=rng)
    intercepts = sample(ens.intercepts, ens.n_neurons, rng=rng)

    # Build the neurons
    if ens.gain is not None and ens.bias is not None:
        gain = sample(ens.gain, ens.n_neurons, rng=rng)
        bias = sample(ens.bias, ens.n_neurons, rng=rng)
    elif ens.gain is not None or ens.bias is not None:
        # TODO: handle this instead of error
        raise NotImplementedError("gain or bias set for %s, but not both. "
                                  "Solving for one given the other is not "
                                  "implemented yet." % ens)
    else:
        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)

    if isinstance(ens.neuron_type, Direct):
        model.sig[ens]['neuron_in'] = Signal(
            np.zeros(ens.dimensions), name='%s.neuron_in' % ens.label)
        model.sig[ens]['neuron_out'] = model.sig[ens]['neuron_in']
        model.add_op(Reset(model.sig[ens]['neuron_in']))
    else:
        model.sig[ens]['neuron_in'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_in" % ens.label)
        model.sig[ens]['neuron_out'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_out" % ens.label)
        model.add_op(Copy(src=Signal(bias, name="%s.bias" % ens.label),
                          dst=model.sig[ens]['neuron_in']))
        # This adds the neuron's operator and sets other signals
        model.build(ens, ens.neuron_type)

    # Scale the encoders
    if isinstance(ens.neuron_type, Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    model.sig[ens]['encoders'] = Signal(
        scaled_encoders, name="%s.scaled_encoders" % ens.label)

    # Create output signal, using built Neurons
    model.add_op(DotInc(
        model.sig[ens]['encoders'],
        model.sig[ens]['in'],
        model.sig[ens]['neuron_in'],
        tag="%s encoding" % ens.label))

    # Output is neural output
    model.sig[ens]['out'] = model.sig[ens]['neuron_out']

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)
