"""Reference implementation for building a model specified by the API."""

import collections
import logging

import numpy as np

import nengo.api
import nengo.util
from nengo.decoders import sample_hypersphere
from nengo.core import ShapeMismatch, Signal, Copy, DotInc, ProdUpdate, \
    Reset, SimPyFunc, SimLIF, SimLIFRate, PythonFunction

logger = logging.getLogger(__name__)

_buildstate_func_dict = {}  # Nengo object -> builder method; set by @builds


class BuiltModel(object):
    """Output of the Builder, used by the Simulator."""

    def __init__(self, dt, label="Model", seed=None):
        self.operators = []
        self.probes = []
        self.sig_in = {}
        self.sig_out = {}

        self.dt = dt
        self.label = label
        self.seed = seed

    def __str__(self):
        return "Model: %s" % self.label


class NeuronsBuildState(object):
    """Encapsulates the state associated with building Neurons."""

    def __init__(self, eval_points, gain, bias, encoders):
        self.eval_points = eval_points
        self.gain = gain
        self.bias = bias
        self.encoders = encoders


class Builder(object):
    """Takes a Model object and returns a BuiltModel.

    This determines the signals and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

      1. Ensembles, Nodes, Neurons, Probes
      2. Networks/Models (recursive)
      3. Connections
    """

    # A decorator that registers the given Nengo object class with the function
    builds = lambda cls: nengo.util.register(
        lambda f: _buildstate_func_dict.__setitem__(cls, f))

    def __init__(self, model, dt):
        # Artifacts of the build process. Needed only in the build scope.
        self._has_built = set()
        self._neurons_state = {}

        # Resources used by the build process.
        seed = nengo.util.random_maxint(np.random) if model.seed is None \
            else model.seed
        self._rng = np.random.RandomState(seed)

        # Build the entire model into output attribute.
        self.output = BuiltModel(dt, "%s, dt=%f" % (model.label, dt), seed)
        self.build(model)

    def has_built(self, obj):
        """Returns true iff obj has been processed by build."""
        return obj in self._has_built

    def mark_built(self, obj):
        """Marks that obj has been processed by build."""
        self._has_built.add(obj)

    def get_neurons_state(self, neurons):
        """Retrieves the NeuronsBuildState for the given neurons."""
        return self._neurons_state[neurons]

    def set_neurons_state(self, neurons, state):
        """Stores a NeuronsBuildState for the given neurons."""
        self._neurons_state[neurons] = state

    def next_seed(self):
        """Yields a seed to use for RNG during build computations."""
        return nengo.util.random_maxint(self._rng)

    def build(self, obj, *args, **kwargs):
        """Builds the given object with the associated builder method."""
        if isinstance(obj, nengo.api.Model):
            # Allow building arbitrary subclasses of Model
            cls = nengo.api.Model
        else:
            cls = obj.__class__

        if not _buildstate_func_dict.has_key(cls):
            raise ValueError("Cannot build object of type '%s'." %
                obj.__class__.__name__)

        if not self.has_built(obj):
            _buildstate_func_dict[cls](self, obj, *args, **kwargs)
            self.mark_built(obj)
        else:
            # This means the Model object contained two objects with the same
            # id, which gives undefined behaviour. This is most likely the
            # result of Neurons being used in two different Ensembles, in which
            # case the same neuron would need two different tuning curves.
            # TODO: Prevent this at pre-build validation time.
            logger.warning("Object (%s) with id=%d has been referenced twice "
                           "within the model.", obj.__class__.__name__, id(obj))

    @builds(nengo.api.Model)
    def _build_model(self, model):
        # 1. Build ensembles and nodes
        logger.info("Building ensembles and nodes")
        for obj in model.ensembles + model.nodes:
            self.build(obj)

        # 2. Then networks
        logger.info("Building networks")
        for network in model.models:
            self.build(network)

        # 3. Then connections
        logger.info("Building connections")
        for conn in model.connections:
            self.build(conn)

    @builds(nengo.api.Ensemble)
    def _build_ensemble(self, ens):
        # Create random number generator
        seed = self.next_seed() if ens.seed is None else ens.seed
        rng = np.random.RandomState(seed)

        # Generate eval points
        eval_points = ens.eval_points
        if eval_points is None:
            eval_points = sample_hypersphere(
                ens.dimensions, ens.EVAL_POINTS, rng) * ens.radius

        # Set up signal
        self.output.sig_in[ens] = Signal(np.zeros(ens.dimensions),
                                         name="%s.signal" % ens.label)
        self.output.operators.append(Reset(self.output.sig_in[ens]))

        # Determine gain (alpha) and j_bias
        if isinstance(ens.max_rates, nengo.api.Sampler):
            max_rates = ens.max_rates.sample(
                ens.neurons.n_neurons, rng=rng)
        else:
            max_rates = ens.max_rates
        if isinstance(ens.intercepts, nengo.api.Sampler):
            intercepts = ens.intercepts.sample(
                ens.neurons.n_neurons, rng=rng)
        else:
            intercepts = ens.intercepts
        gain, bias = ens.neurons.gain_bias(max_rates, intercepts)

        # Set up encoders
        if ens.encoders is None:
            if isinstance(ens.neurons, nengo.api.Direct):
                encoders = np.identity(ens.dimensions)
            else:
                encoders = sample_hypersphere(
                    ens.dimensions, ens.neurons.n_neurons, rng, surface=True)
        else:
            encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions); "
                    "in this case %s." % (encoders.shape, enc_shape))
            norm = np.sum(encoders * encoders, axis=1)[:, np.newaxis]
            encoders /= np.sqrt(norm)

        # Store the values that we need to recall for Connection/Neuron building
        self.set_neurons_state(
            ens.neurons, NeuronsBuildState(eval_points, gain, bias, encoders))

        # Build the neurons
        self.build(ens.neurons, bias, ens.dimensions)

        # Scale the encoders
        if isinstance(ens.neurons, nengo.api.Direct):
            scaled_encoders = encoders
        else:
            scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

        # Create output signal, using built Neurons
        self.output.operators.append(DotInc(
            Signal(scaled_encoders, name="%s.scaled_encoders" % ens.label),
            self.output.sig_in[ens],
            self.output.sig_in[ens.neurons],
            tag="%s encoding" % ens.label))

        # Output is neural output
        self.output.sig_out[ens] = self.output.sig_out[ens.neurons]

        # Build the probes
        for probe in ens.probes["decoded_output"]:
            self.build(probe, dimensions=ens.dimensions)
        for probe in ens.probes["spikes"] + ens.probes["voltages"]:
            self.build(probe, dimensions=ens.neurons.n_neurons)

    def _build_pyfunc(self, pyfn):
        if pyfn.n_in > 0:
            self.output.sig_in[pyfn] = Signal(np.zeros(pyfn.n_in),
                                              name="%s.input" % pyfn.label)
            self.output.operators.append(Reset(self.output.sig_in[pyfn]))
        self.output.sig_out[pyfn] = Signal(np.zeros(pyfn.n_out),
                                           name="%s.output" % pyfn.label)
        self.output.operators.append(
            SimPyFunc(output=self.output.sig_out[pyfn],
                      J=self.output.sig_in[pyfn] if pyfn.n_in > 0 else None,
                      fn=pyfn.fn,
                      n_args=pyfn.n_args))

    @builds(nengo.api.Node)
    def _build_node(self, node):
        # Get input
        if (node.output is None
                or isinstance(node.output, collections.Callable)):
            if node.size_in > 0:
                self.output.sig_in[node] = Signal(
                    np.zeros(node.size_in), name="%s.signal" % node.label)
                # Reset input signal to 0 each timestep
                self.output.operators.append(Reset(self.output.sig_in[node]))

        # Provide output
        if node.output is None:
            self.output.sig_out[node] = self.output.sig_in[node]
        elif not isinstance(node.output, collections.Callable):
            self.output.sig_out[node] = Signal(node.output, name=node.label)
        else:
            pyfn = PythonFunction(fn=node.output,
                                  n_in=node.size_in,
                                  n_out=node.size_out,
                                  label="%s.pyfn" % node.label)
            self._build_pyfunc(pyfn)
            if node.size_in > 0:
                self.output.operators.append(DotInc(
                    self.output.sig_in[node],
                    Signal(1.0, name="1"),
                    self.output.sig_in[pyfn],
                    tag="%s input" % node.label))
            self.output.sig_out[node] = self.output.sig_out[pyfn]

        # Set up probes
        for probe in node.probes["output"]:
            self.build(probe, dimensions=self.output.sig_out[node].shape)

    @builds(nengo.api.Probe)
    def _build_probe(self, probe, dimensions):
        probe_signal = Signal(np.zeros(dimensions), name=probe.label)
        self.output.sig_in[probe] = probe_signal
        # Reset input signal to 0 each timestep
        self.output.operators.append(Reset(probe_signal))
        self.output.probes.append(probe)

    @classmethod
    def _filter_coefs(cls, pstc, dt):
        pstc = max(pstc, dt)
        decay = np.exp(-dt / pstc)
        return decay, (1.0 - decay)

    def _filtered_signal(self, signal, pstc):
        name = "%s.filtered(%f)" % (signal.name, pstc)
        filtered = Signal(np.zeros(signal.size), name=name)
        o_coef, n_coef = self._filter_coefs(pstc=pstc, dt=self.output.dt)
        self.output.operators.append(ProdUpdate(
            Signal(n_coef, name="n_coef"),
            signal,
            Signal(o_coef, name="o_coef"),
            filtered,
            tag="%s filtering" % name))
        return filtered

    def _direct_pyfunc(self, input_signal, function, n_out, label):
        pyfunc = PythonFunction(
            fn=function, n_in=input_signal.size, n_out=n_out, label=label)
        self._build_pyfunc(pyfunc)
        self.output.operators.append(DotInc(
            input_signal,
            Signal(1.0, name="1"),
            self.output.sig_in[pyfunc],
            tag="%s input" % label))
        return pyfunc

    @builds(nengo.api.Connection)
    def _build_connection(self, conn):
        rng = np.random.RandomState(self.next_seed())

        self.output.sig_in[conn] = self.output.sig_out[conn.pre]
        self.output.sig_out[conn] = self.output.sig_in[conn.post]
        if conn.modulatory:
            # Make a new signal, effectively detaching from post
            self.output.sig_out[conn] = Signal(
                np.zeros(self.output.sig_out[conn].size),
                name="%s.mod_output" % conn.label)
            # Add reset operator?
            # XXX add unit test

        # Figure out the signal going across this connection
        if (isinstance(conn.pre, nengo.api.Ensemble)
                and isinstance(conn.pre.neurons, nengo.api.Direct)):
            # 1. Decoded connection in directmode
            if conn.function is None:
                signal = self.output.sig_in[conn]
            else:
                pyfunc = self._direct_pyfunc(
                    self.output.sig_in[conn],
                    lambda t, x: conn.function(x),
                    conn.dimensions,
                    conn.label)
                signal = self.output.sig_out[pyfunc]

            if conn.filter is not None and conn.filter > self.output.dt:
                signal = self._filtered_signal(signal, conn.filter)
        elif isinstance(conn.pre, nengo.api.Ensemble):
            # 2. Normal decoded connection
            pre_state = self.get_neurons_state(conn.pre.neurons)

            # Use Connection's eval_points, or built Ensemble's eval_points
            eval_points = conn.eval_points if pre_state.eval_points is None \
                else pre_state.eval_points
            eval_points = np.array(eval_points, dtype=np.float64)
            if eval_points.ndim == 1:
                eval_points.shape = (-1, 1)

            signal = Signal(np.zeros(conn.dimensions), name=conn.label)

            x = np.dot(eval_points, pre_state.encoders.T / conn.pre.radius)
            activities = self.output.dt * conn.pre.neurons.rates(
                x, pre_state.gain, pre_state.bias)
            if conn.function is None:
                targets = eval_points
            else:
                targets = np.array([conn.function(ep) for ep in eval_points])
                if targets.ndim < 2:
                    targets.shape = targets.shape[0], 1
            decoders = conn.decoder_solver(activities, targets, rng).T

            if conn.filter is not None and conn.filter > self.output.dt:
                o_coef, n_coef = self._filter_coefs(
                    pstc=conn.filter, dt=self.output.dt)
                decoder_signal = Signal(
                    decoders * n_coef, name="%s.decoders * n_coef" % conn.label)
            else:
                o_coef = 0
                decoder_signal = Signal(
                    decoders, name="%s.decoders" % conn.label)
            self.output.operators.append(ProdUpdate(
                decoder_signal,
                self.output.sig_in[conn],
                Signal(o_coef, name="o_coef"),
                signal,
                tag="%s decoding" % conn.label))
        elif conn.filter is not None and conn.filter > self.output.dt:
            # 3. Filtered connection
            signal = self._filtered_signal(
                self.output.sig_in[conn], conn.filter)
        else:
            # 4. Direct connection
            signal = self.output.sig_in[conn]

        # Set up transform
        transform = np.asarray(conn.transform, dtype=np.float64)
        if isinstance(conn.post, nengo.api.Neurons):
            if not self.has_built(conn.post):
                # Since it hasn't been built, it wasn't added to the model,
                # which is most likely because the Neurons weren't associated
                # with an Ensemble.
                raise RuntimeError("Connection '%s' refers to Neurons '%s' "
                                     "that are not a part of any Ensemble." % (
                                     conn, conn.post))
            transform *= self.get_neurons_state(conn.post).gain[:, np.newaxis]

        self.output.operators.append(
            DotInc(Signal(transform, name="%s.transform" % conn.label),
                   signal,
                   self.output.sig_out[conn],
                   tag=conn.label))

        # Set up probes
        for _ in conn.probes["signal"]:
            logger.error("Connection probes not yet implemented.")

    @builds(nengo.api.Direct)
    def _build_direct(self, direct, bias, dimensions):
        assert bias is None
        self.output.sig_in[direct] = Signal(np.zeros(dimensions),
                                            name=direct.label)
        self.output.sig_out[direct] = self.output.sig_in[direct]
        self.output.operators.append(Reset(self.output.sig_in[direct]))

    def _build_neurons(self, neurons, bias):
        self.output.sig_in[neurons] = Signal(
            np.zeros(neurons.n_neurons), name="%s.input" % neurons.label)
        self.output.sig_out[neurons] = Signal(
            np.zeros(neurons.n_neurons), name="%s.output" % neurons.label)

        self.output.operators.append(Copy(
            src=Signal(bias, name="%s.bias" % neurons.label),
            dst=self.output.sig_in[neurons]))

        # Set up probes
        for probe in neurons.probes["output"]:
            self.build(probe, dimensions=neurons.n_neurons)

    @builds(nengo.api.LIFRate)
    def _build_lifrate(self, lif, bias, dummy_dimensions):
        if lif.n_neurons <= 0:
            raise ValueError(
                "Number of neurons (%d) must be positive." % lif.n_neurons)
        self._build_neurons(lif, bias)
        self.output.operators.append(SimLIFRate(
            output=self.output.sig_out[lif], J=self.output.sig_in[lif], nl=lif))

    @builds(nengo.api.LIF)
    def _build_lif(self, lif, bias, dummy_dimensions):
        if lif.n_neurons <= 0:
            raise ValueError(
                "Number of neurons (%d) must be positive." % lif.n_neurons)
        self._build_neurons(lif, bias)
        voltage = Signal(np.zeros(lif.n_neurons), name="%s.voltage" % lif.label)
        refractory_time = Signal(np.zeros(lif.n_neurons),
                                 name="%s.refractory_time" % lif.label)
        self.output.operators.append(SimLIF(output=self.output.sig_out[lif],
                                            J=self.output.sig_in[lif],
                                            nl=lif,
                                            voltage=voltage,
                                     refractory_time=refractory_time))
