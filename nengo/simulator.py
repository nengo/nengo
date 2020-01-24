"""Reference simulator for nengo models.

This backend is relatively fast, and works on general purpose computers.
Other Nengo backends provide more specialized Simulators for custom platforms.
"""

import logging
import warnings
from collections.abc import Mapping

import numpy as np

import nengo.utils.numpy as npext
from nengo.builder import Model
from nengo.builder.optimizer import optimize as opmerge_optimize
from nengo.builder.signal import SignalDict
from nengo.cache import get_default_decoder_cache
from nengo.exceptions import ReadonlyError, SimulatorClosed, ValidationError
from nengo.utils.graphs import toposort
from nengo.utils.progress import Progress, ProgressTracker
from nengo.utils.simulator import operator_dependency_graph

logger = logging.getLogger(__name__)


class Simulator:
    """Reference simulator for Nengo models.

    The simulator takes a `.Network` and builds internal data structures to
    run the model defined by that network. Run the simulator with the
    `~.Simulator.run` method, and access probed data through the
    ``data`` attribute.

    Building and running the simulation may allocate resources like files
    and sockets. To properly free these resources, call the `.Simulator.close`
    method. Alternatively, `.Simulator.close` will automatically be called
    if you use the ``with`` syntax:

    .. testcode::

       with nengo.Network() as my_network:
           my_ensemble = nengo.Ensemble(10, 1)
           my_probe = nengo.Probe(my_ensemble)

       with nengo.Simulator(my_network) as sim:
           sim.run(0.1)
       print(sim.data[my_probe])

    .. testoutput::
       :hide:

       ...

    Note that the ``data`` attribute is still accessible even when a simulator
    has been closed. Running the simulator, however, will raise an error.

    Parameters
    ----------
    network : Network or None
        A network object to be built and then simulated. If None,
        then a `.Model` with the build model must be provided instead.
    dt : float, optional
        The length of a simulator timestep, in seconds.
    seed : int, optional
        A seed for all stochastic operators used in this simulator.
        Will be set to ``network.seed + 1`` if not given.
    model : Model, optional
        A `.Model` that contains build artifacts to be simulated.
        Usually the simulator will build this model for you; however, if you
        want to build the network manually, or you want to inject build
        artifacts in the model before building the network, then you can
        pass in a `.Model` instance.
    progress_bar : bool or ProgressBar, optional
        Progress bar for displaying build and simulation progress.

        If ``True``, the default progress bar will be used.
        If ``False``, the progress bar will be disabled.
        For more control over the progress bar, pass in a ``ProgressBar``
        instance.
    optimize : bool, optional
        If ``True``, the builder will run an additional optimization step
        that can speed up simulations significantly at the cost of slower
        builds. If running models for very small amounts of time,
        pass ``False`` to disable the optimizer.

    Attributes
    ----------
    closed : bool
        Whether the simulator has been closed.
        Once closed, it cannot be reopened.
    data : SimulationData
        The `.SimulationData` mapping from Nengo objects to the data associated
        with those objects. In particular, each `.Probe` maps to the data
        probed while running the simulation.
    dg : dict
        A dependency graph mapping from each `.Operator` to the operators
        that depend on that operator.
    model : Model
        The `.Model` containing the signals and operators necessary to
        simulate the network.
    signals : SignalDict
        The `.SignalDict` mapping from `.Signal` instances to NumPy arrays.

    """

    def __init__(
        self, network, dt=0.001, seed=None, model=None, progress_bar=True, optimize=True
    ):
        self.closed = True  # Start closed in case constructor raises exception
        self.progress_bar = progress_bar
        self.optimize = optimize

        if model is None:
            self.model = Model(
                dt=float(dt),
                label="%s, dt=%f" % (network, dt),
                decoder_cache=get_default_decoder_cache(),
            )
        else:
            self.model = model

        pt = ProgressTracker(progress_bar, Progress("Building", "Build"))
        with pt:
            if network is not None:
                # Build the network into the model
                self.model.build(network, progress=pt.next_stage("Building", "Build"))

            # Order the steps (they are made in `Simulator.reset`)
            self.dg = operator_dependency_graph(self.model.operators)

            if optimize:
                with pt.next_stage("Building (running optimizer)", "Optimization"):
                    opmerge_optimize(self.model, self.dg)

        self._step_order = [op for op in toposort(self.dg) if hasattr(op, "make_step")]

        # -- map from Signal.base -> ndarray
        self.signals = SignalDict()
        for op in self.model.operators:
            op.init_signals(self.signals)

        # Add built states to the raw simulation data dictionary
        self._sim_data = self.model.params

        # Provide a nicer interface to simulation data
        self.data = SimulationData(self._sim_data)

        if seed is None:
            if network is not None and network.seed is not None:
                seed = network.seed + 1
            else:
                seed = np.random.randint(npext.maxint)

        self.closed = False
        self.reset(seed=seed)

    def __del__(self):
        """Raise a ResourceWarning if we are deallocated while open."""
        if not self.closed:
            warnings.warn(
                "Simulator with model=%s was deallocated while open. Please "
                "close simulators manually to ensure resources are properly "
                "freed." % self.model,
                ResourceWarning,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getstate__(self):
        signals = (
            {k: v for k, v in self.signals.items() if not k.readonly}
            if self.signals is not None
            else {}
        )
        probe_outputs = {probe: self._sim_data[probe] for probe in self.model.probes}
        return dict(
            model=self.model,
            signals=signals,
            probe_outputs=probe_outputs,
            dt=self.dt,
            seed=self.seed,
            progress_bar=self.progress_bar,
            optimize=self.optimize,
            closed=self.closed,
        )

    def __setstate__(self, state):
        self.__init__(
            network=None,
            model=state["model"],
            dt=state["dt"],
            seed=state["seed"],
            progress_bar=state["progress_bar"],
            optimize=state["optimize"],
        )
        for key, value in state["signals"].items():
            self.signals[key] = value
        for key, value in state["probe_outputs"].items():
            self._sim_data[key].extend(value)
        if state["closed"]:
            self.close()

    @property
    def dt(self):
        """(float) The time step of the simulator."""
        return self.model.dt

    @dt.setter
    def dt(self, dummy):
        raise ReadonlyError(attr="dt", obj=self)

    @property
    def n_steps(self):
        """(int) The current time step of the simulator."""
        return self._n_steps

    @property
    def time(self):
        """(float) The current time of the simulator."""
        return self._time

    def clear_probes(self):
        """Clear all probe histories.

        .. versionadded:: 3.0.0
        """
        for probe in self.model.probes:
            self._sim_data[probe] = []
        self.data.reset()  # clear probe cache

    def close(self):
        """Closes the simulator.

        Any call to `.Simulator.run`, `.Simulator.run_steps`,
        `.Simulator.step`, and `.Simulator.reset` on a closed simulator raises
        a `.SimulatorClosed` exception.
        """
        self.closed = True
        self.signals = None  # signals may no longer exist on some backends

    def _probe(self):
        """Copy all probed signals to buffers."""
        self._probe_step_time()

        for probe in self.model.probes:
            period = 1 if probe.sample_every is None else probe.sample_every / self.dt
            if self.n_steps % period < 1:
                tmp = self.signals[self.model.sig[probe]["in"]].copy()
                self._sim_data[probe].append(tmp)

    def _probe_step_time(self):
        self._n_steps = self.signals[self.model.step].item()
        self._time = self.signals[self.model.time].item()

    def reset(self, seed=None):
        """Reset the simulator state.

        Parameters
        ----------
        seed : int, optional
            A seed for all stochastic operators used in the simulator.
            This will change the random sequences generated for noise
            or inputs (e.g. from processes), but not the built objects
            (e.g. ensembles, connections).
        """
        if self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        if seed is not None:
            self.seed = seed

        # reset signals
        for key in self.signals:
            self.signals.reset(key)

        # rebuild steps (resets ops with their own state, like Processes)
        self.rng = np.random.RandomState(self.seed)
        self._steps = [
            op.make_step(self.signals, self.dt, self.rng) for op in self._step_order
        ]

        self.clear_probes()

        self._probe_step_time()

    def run(self, time_in_seconds, progress_bar=None):
        """Simulate for the given length of time.

        If the given length of time is not a multiple of ``dt``,
        it will be rounded to the nearest ``dt``. For example, if ``dt``
        is 0.001 and ``run`` is called with ``time_in_seconds=0.0006``,
        the simulator will advance one timestep, resulting in the actual
        simulator time being 0.001.

        The given length of time must be positive. The simulator cannot
        be run backwards.

        Parameters
        ----------
        time_in_seconds : float
            Amount of time to run the simulation for. Must be positive.
        progress_bar : bool or ProgressBar, optional
            Progress bar for displaying the progress of the simulation run.

            If True, the default progress bar will be used.
            If False, the progress bar will be disabled.
            For more control over the progress bar, pass in a ``ProgressBar``
            instance.
        """
        if time_in_seconds < 0:
            raise ValidationError(
                "Must be positive (got %g)" % (time_in_seconds,), attr="time_in_seconds"
            )

        steps = int(np.round(float(time_in_seconds) / self.dt))

        if steps == 0:
            warnings.warn(
                "%g results in running for 0 timesteps. Simulator "
                "still at time %g." % (time_in_seconds, self.time)
            )
        else:
            logger.info(
                "Running %s for %f seconds, or %d steps",
                self.model.label,
                time_in_seconds,
                steps,
            )
            self.run_steps(steps, progress_bar=progress_bar)

    def run_steps(self, steps, progress_bar=None):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        progress_bar : bool or ProgressBar, optional
            Progress bar for displaying the progress of the simulation run.

            If True, the default progress bar will be used.
            If False, the progress bar will be disabled.
            For more control over the progress bar, pass in a ``ProgressBar``
            instance.
        """
        if progress_bar is None:
            progress_bar = self.progress_bar

        with ProgressTracker(
            progress_bar, Progress("Simulating", "Simulation", steps)
        ) as pt:
            for i in range(steps):
                self.step()
                pt.total_progress.step()

    def step(self):
        """Advance the simulator by 1 step (``dt`` seconds)."""
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        old_err = np.seterr(invalid="raise", divide="ignore")
        try:
            for step_fn in self._steps:
                step_fn()
        finally:
            np.seterr(**old_err)

        self._probe()

    def trange(self, dt=None, sample_every=None):
        """Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        sample_every : float, optional
            The sampling period of the probe to create a range for.
            If None, a time value for every ``dt`` will be produced.

            .. versionchanged:: 3.0.0
               Renamed from dt to sample_every
        """
        if dt is not None:
            if sample_every is not None:
                raise ValidationError(
                    "Cannot specify both `dt` and `sample_every`. "
                    "Use `sample_every` only.",
                    attr="dt",
                    obj=self,
                )
            warnings.warn(
                "`dt` is deprecated. Use `sample_every` instead.", DeprecationWarning
            )
            sample_every = dt
        period = 1 if sample_every is None else sample_every / self.dt
        steps = np.arange(1, self.n_steps + 1)
        return self.dt * steps[steps % period < 1]


class SimulationData(Mapping):
    """Data structure used to access simulation data from the model.

    The main use case for this is to access Probe data; for example,
    ``probe_data = sim.data[my_probe]``. However, it is also used to access the
    parameters of objects in the model; for example, encoder values for an ensemble
    can be accessed via ``encoders = sim.data[my_ens].encoders``.

    This is like a view on the raw simulation data manipulated by the Simulator,
    which allows the raw simulation data to be optimized for speed while this
    provides a more user-friendly interface.

    .. versionchanged:: 3.0.0
       Renamed from ProbeDict to SimulationData
    """

    def __init__(self, raw):
        super().__init__()
        self.raw = raw
        self._cache = {}

    def __getitem__(self, key):
        """Return simulation data for ``key`` object.

        For speed reasons, the simulator uses Python lists for Probe data
        and we want to return NumPy arrays.
        """
        if key not in self._cache or len(self._cache[key]) != len(self.raw[key]):
            rval = self.raw[key]
            if isinstance(rval, list):
                rval = np.asarray(rval)
                rval.setflags(write=False)
            self._cache[key] = rval
        return self._cache[key]

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)

    def __repr__(self):
        return repr(self.raw)

    def __str__(self):
        return str(self.raw)

    def reset(self):
        self._cache.clear()


class ProbeDict(SimulationData):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ProbeDict has been renamed to SimulationData. This alias "
            "will be removed in Nengo 3.1."
        )
        super().__init__(*args, **kwargs)
