import collections
import warnings

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder import Builder, Signal
from nengo.builder.operator import Copy, DotInc, Reset
from nengo.dists import Distribution, get_samples
from nengo.ensemble import Ensemble
from nengo.exceptions import BuildError, NengoWarning
from nengo.neurons import Direct
from nengo.rc import rc
from nengo.utils.builder import default_n_eval_points

built_attrs = ['eval_points',
               'encoders',
               'intercepts',
               'max_rates',
               'scaled_encoders',
               'gain',
               'bias']


class BuiltEnsemble(collections.namedtuple('BuiltEnsemble', built_attrs)):
    """Collects the parameters generated in `.build_ensemble`.

    These are stored here because in the majority of cases the equivalent
    attribute in the original ensemble is a `.Distribution`. The attributes
    of a BuiltEnsemble are the full NumPy arrays used in the simulation.

    See the `.Ensemble` documentation for more details on each parameter.

    Parameters
    ----------
    eval_points : ndarray
        Evaluation points.
    encoders : ndarray
        Normalized encoders.
    intercepts : ndarray
        X-intercept of each neuron.
    max_rates : ndarray
        Maximum firing rates for each neuron.
    scaled_encoders : ndarray
        Normalized encoders scaled by the gain and radius.
        This quantity is used in the actual simulation, unlike ``encoders``.
    gain : ndarray
        Gain of each neuron.
    bias : ndarray
        Bias current injected into each neuron.
    """

    __slots__ = ()

    def __new__(cls, eval_points, encoders, intercepts, max_rates,
                scaled_encoders, gain, bias):
        # Overridden to suppress the default __new__ docstring
        return tuple.__new__(cls, (eval_points, encoders, intercepts,
                                   max_rates, scaled_encoders, gain, bias))


def gen_eval_points(
        ens, eval_points, rng, scale_eval_points=True, dtype=None):
    dtype = rc.float_dtype if dtype is None else dtype
    if isinstance(eval_points, Distribution):
        n_points = ens.n_eval_points
        if n_points is None:
            n_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
        eval_points = eval_points.sample(n_points, ens.dimensions, rng)
        eval_points = eval_points.astype(dtype)
    else:
        if (ens.n_eval_points is not None
                and eval_points.shape[0] != ens.n_eval_points):
            warnings.warn("Number of eval_points doesn't match "
                          "n_eval_points. Ignoring n_eval_points.")
        eval_points = np.array(eval_points, dtype=dtype)
        assert eval_points.ndim == 2

    if scale_eval_points:
        eval_points *= ens.radius  # scale by ensemble radius
    return eval_points


def get_activities(built_ens, ens, eval_points):
    x = np.dot(eval_points, built_ens.encoders.T / ens.radius)
    return ens.neuron_type.rates(x, built_ens.gain, built_ens.bias)


def get_gain_bias(ens, rng=np.random, dtype=None):
    dtype = rc.float_dtype if dtype is None else dtype
    if ens.gain is not None and ens.bias is not None:
        gain = get_samples(ens.gain, ens.n_neurons, rng=rng)
        bias = get_samples(ens.bias, ens.n_neurons, rng=rng)
        max_rates, intercepts = ens.neuron_type.max_rates_intercepts(
            gain, bias)

        if (ens.max_rates is not Ensemble.max_rates.default
                or ens.intercepts is not Ensemble.intercepts.default):
            warnings.warn(NengoWarning(
                "Specifying the gains and biases for %s imposes a set of "
                "maximum firing rates and intercepts. Further specifying "
                "either max_rates or intercepts has no effect." % ens))

    elif ens.gain is not None or ens.bias is not None:
        # TODO: handle this instead of error
        raise NotImplementedError("gain or bias set for %s, but not both. "
                                  "Solving for one given the other is not "
                                  "implemented yet." % ens)
    else:
        max_rates = get_samples(ens.max_rates, ens.n_neurons, rng=rng)
        intercepts = get_samples(ens.intercepts, ens.n_neurons, rng=rng)
        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
        if gain is not None and (
                not np.all(np.isfinite(gain)) or np.any(gain <= 0.)):
            raise BuildError(
                "The specified intercepts for %s lead to neurons with "
                "negative or non-finite gain. Please adjust the intercepts so "
                "that all gains are positive. For most neuron types (e.g., "
                "LIF neurons) this is achieved by reducing the maximum "
                "intercept value to below 1." % ens)

    gain = gain.astype(dtype) if gain is not None else gain
    bias = bias.astype(dtype) if bias is not None else bias
    max_rates = max_rates.astype(dtype) if max_rates is not None else max_rates
    intercepts = (intercepts.astype(dtype) if intercepts is not None
                  else intercepts)
    return gain, bias, max_rates, intercepts


@Builder.register(Ensemble)  # noqa: C901
def build_ensemble(model, ens):
    """Builds an `.Ensemble` object into a model.

    A brief summary of what happens in the ensemble build process, in order:

    1. Generate evaluation points and encoders.
    2. Normalize encoders to unit length.
    3. Determine bias and gain.
    4. Create neuron input signal
    5. Add operator for injecting bias.
    6. Call build function for neuron type.
    7. Scale encoders by gain and radius.
    8. Add operators for multiplying decoded input signal by encoders and
       incrementing the result in the neuron input signal.
    9. Call build function for injected noise.

    Some of these steps may be altered or omitted depending on the parameters
    of the ensemble, in particular the neuron type. For example, most steps are
    omitted for the `.Direct` neuron type.

    Parameters
    ----------
    model : Model
        The model to build into.
    ens : Ensemble
        The ensemble to build.

    Notes
    -----
    Sets ``model.params[ens]`` to a `.BuiltEnsemble` instance.
    """

    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    eval_points = gen_eval_points(ens, ens.eval_points,
                                  rng=rng, dtype=rc.float_dtype)

    # Set up signal
    model.sig[ens]['in'] = Signal(shape=ens.dimensions, name="%s.signal" % ens)
    model.add_op(Reset(model.sig[ens]['in']))

    # Set up encoders
    if isinstance(ens.neuron_type, Direct):
        encoders = np.identity(ens.dimensions, dtype=rc.float_dtype)
    elif isinstance(ens.encoders, Distribution):
        encoders = get_samples(
            ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)
        encoders = np.asarray(encoders, dtype=rc.float_dtype)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=rc.float_dtype)
    if ens.normalize_encoders:
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Build the neurons
    gain, bias, max_rates, intercepts = get_gain_bias(ens, rng,
                                                      dtype=rc.float_dtype)

    if isinstance(ens.neuron_type, Direct):
        model.sig[ens.neurons]['in'] = Signal(
            shape=ens.dimensions, name='%s.neuron_in' % ens)
        model.sig[ens.neurons]['out'] = model.sig[ens.neurons]['in']
        model.add_op(Reset(model.sig[ens.neurons]['in']))
    else:
        model.sig[ens.neurons]['in'] = Signal(
            shape=ens.n_neurons, name="%s.neuron_in" % ens)
        model.sig[ens.neurons]['out'] = Signal(
            shape=ens.n_neurons, name="%s.neuron_out" % ens)
        model.sig[ens.neurons]['bias'] = Signal(
            bias, name="%s.bias" % ens, readonly=True)
        model.add_op(Copy(model.sig[ens.neurons]['bias'],
                          model.sig[ens.neurons]['in']))
        # This adds the neuron's operator and sets other signals
        model.build(ens.neuron_type, ens.neurons)

    # Scale the encoders
    if isinstance(ens.neuron_type, Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    model.sig[ens]['encoders'] = Signal(
        scaled_encoders, name="%s.scaled_encoders" % ens, readonly=True)

    # Inject noise if specified
    if ens.noise is not None:
        model.build(ens.noise, sig_out=model.sig[ens.neurons]['in'],
                    mode="inc")

    # Create output signal, using built Neurons
    model.add_op(DotInc(
        model.sig[ens]['encoders'],
        model.sig[ens]['in'],
        model.sig[ens.neurons]['in'],
        tag="%s encoding" % ens))

    # Output is neural output
    model.sig[ens]['out'] = model.sig[ens.neurons]['out']

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)
