import warnings

import numpy as np

from nengo.base import Process
from nengo.exceptions import ValidationError
from nengo.linear_system import LinearSystem
from nengo.params import IntParam, NumberParam, Parameter, Unconfigurable
from nengo.rc import rc
from nengo.utils.numpy import as_shape, is_number


def _is_empty(M):
    """Determine whether a matrix is size zero or all zeros"""
    return M.size == 0 or (M == 0).all()


class Synapse(Process):
    """Abstract base class for synapse models.

    Conceptually, a synapse model emulates a biological synapse, taking in
    input in the form of released neurotransmitter and opening ion channels
    to allow more or less current to flow into the neuron.

    In Nengo, the implementation of a synapse is as a specific case of a
    `.Process` in which the input and output shapes are the same.
    The input is the current across the synapse, and the output is the current
    that will be induced in the postsynaptic neuron.

    Synapses also contain the `.Synapse.filt` and `.Synapse.filtfilt` methods,
    which make it easy to use Nengo's synapse models outside of Nengo
    simulations.

    Parameters
    ----------
    default_size_in : int, optional
        The size_in used if not specified.
    default_size_out : int, optional
        The size_out used if not specified.
        If None, will be the same as default_size_in.
    default_dt : float, optional
        The simulation timestep used if not specified.
    seed : int, optional
        Random number seed. Ensures random factors will be the same each run.

    Attributes
    ----------
    default_dt : float
        The simulation timestep used if not specified.
    default_size_in : int
        The size_in used if not specified.
    default_size_out : int
        The size_out used if not specified.
    seed : int, optional
        Random number seed. Ensures random factors will be the same each run.
    """

    def __init__(
        self, default_size_in=1, default_size_out=None, default_dt=0.001, seed=None
    ):
        if default_size_out is None:
            default_size_out = default_size_in
        super().__init__(
            default_size_in=default_size_in,
            default_size_out=default_size_out,
            default_dt=default_dt,
            seed=seed,
        )

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        raise NotImplementedError("Synapse must implement make_state")

    def filt(self, x, dt=None, axis=0, y0=0, copy=True, filtfilt=False):
        """Filter ``x`` with this synapse model.

        Parameters
        ----------
        x : array_like
            The signal to filter.
        dt : float, optional
            The timestep of the input signal.
            If None, ``default_dt`` will be used.
        axis : int, optional
            The axis along which to filter.
        y0 : array_like, optional
            The starting state of the filter output. Must be zero for
            unstable linear systems.
        copy : bool, optional
            Whether to copy the input data, or simply work in-place.
        filtfilt : bool, optional
            If True, runs the process forward then backward on the signal,
            for zero-phase filtering (like Matlab's ``filtfilt``).
        """
        # This function is very similar to `Process.apply`, but allows for
        # a) filtering along any axis, and b) zero-phase filtering (filtfilt).
        dt = self.default_dt if dt is None else dt
        filtered = np.array(x, copy=copy, dtype=rc.float_dtype)
        filt_view = np.rollaxis(filtered, axis=axis)  # rolled view on filtered

        shape_in = shape_out = as_shape(filt_view[0].shape, min_dim=1)
        state = self.make_state(shape_in, shape_out, dt, dtype=filtered.dtype, y0=y0)
        step = self.make_step(shape_in, shape_out, dt, rng=None, state=state)

        for i, signal_in in enumerate(filt_view):
            filt_view[i] = step(i * dt, signal_in)

        if filtfilt:  # Flip the filt_view and filter again
            n = len(filt_view) - 1
            filt_view = filt_view[::-1]
            for i, signal_in in enumerate(filt_view):
                filt_view[i] = step((n - i) * dt, signal_in)

        return filtered

    def filtfilt(self, x, **kwargs):
        """Zero-phase filtering of ``x`` using this filter.

        Equivalent to `filt(x, filtfilt=True, **kwargs) <.Synapse.filt>`.
        """
        return self.filt(x, filtfilt=True, **kwargs)


class LinearFilter(LinearSystem, Synapse):
    """General linear time-invariant (LTI) system synapse.

    This class can be used to implement any linear filter, given a description of the
    filter. [1]_

    Parameters
    ----------
    sys : tuple
        A tuple describing the linear system used for filtering, in
        ``(numerator, denominator)``, ``(zeros, poles, gain)``, or ``(A, B, C, D)``
        (state space) form. See `.LinearSystem` for more details on these forms.
        The system must have both an input size and output size of 1.
    den : array_like, optional
        Denominator coefficients of the transfer function. If used, the first (``sys``)
        argument is treated as the numerator coefficients of the transfer function.
        This way of specifying transfer functions is deprecated; the preferred method
        is to pass a ``(num, den)`` tuple to the ``sys`` argument.

        .. deprecated:: 3.1.0
    analog : boolean, optional
        Whether the synapse coefficients are analog (i.e. continuous-time),
        or discrete. Analog coefficients will be converted to discrete for
        simulation using the simulator ``dt``.
    method : string, optional
        The method to use for discretization (if ``analog`` is True). See
        `scipy.signal.cont2discrete` for information about the options.

        .. versionadded:: 3.0.0
    x0 : array_like, optional
        Initial values for the system state. The last dimension must equal the
        ``state_size``.

        .. versionadded:: 3.1.0
    initial_output : Distribution or float or (n_synapses,) array_like, optional
        Initial output value(s) represented by the synapses. ``n_synapses`` is typically
        equal to the connection's ``size_out``, except when ``post_obj`` is an
        ``Ensemble`` and ``solver.weights`` is set, in which case ``n_synapses`` equals
        the number of neurons in the post ``Ensemble``.

        .. versionadded:: 3.1.0

    Attributes
    ----------
    analog : boolean
        Whether the synapse coefficients are analog (i.e. continuous-time),
        or discrete. Analog coefficients will be converted to discrete for
        simulation using the simulator ``dt``.
    method : string
        The method to use for discretization (if ``analog`` is True). See
        `scipy.signal.cont2discrete` for information about the options.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Filter_%28signal_processing%29
    """

    _argrepr_filter = {"den"}

    def __init__(
        self, sys, den=None, analog=True, method="zoh", x0=0, default_dt=0.001
    ):
        if den is not None:
            warnings.warn(
                DeprecationWarning(
                    "`den` is deprecated. Pass systems in transfer function form as a "
                    "`(numerator, denominator)` 2-tuple instead."
                )
            )
            sys = (sys, den)

        super().__init__(
            sys,
            analog=analog,
            method=method,
            x0=x0,
            default_dt=default_dt,
        )

        if self.input_size != 1:
            raise ValidationError(
                "LinearFilter systems can only have one input dimension "
                f"(got {self.input_size})",
                attr="input_size",
                obj=self,
            )

        if self.output_size != 1:
            raise ValidationError(
                "LinearFilter systems can only have one output dimension "
                f"(got {self.output_size})",
                attr="output_size",
                obj=self,
            )

        self._factory = LinearFilter

    def evaluate(self, frequencies):
        """Evaluate the transfer function at the given frequencies.

        Examples
        --------
        Using the ``evaluate`` function to make a Bode plot:

        .. testcode::

           synapse = nengo.synapses.LinearFilter(([1], [0.02, 1]))
           f = np.logspace(-1, 3, 100)
           y = synapse.evaluate(f)
           plt.subplot(211); plt.semilogx(f, 20*np.log10(np.abs(y)))
           plt.xlabel('frequency [Hz]'); plt.ylabel('magnitude [dB]')
           plt.subplot(212); plt.semilogx(f, np.angle(y))
           plt.xlabel('frequency [Hz]'); plt.ylabel('phase [radians]')
        """
        frequencies = 2.0j * np.pi * np.asarray(frequencies)
        w = frequencies if self.analog else np.exp(frequencies)
        num, den = self.tf
        assert num.ndim == 2 and num.shape[0] == 1
        y = np.polyval(num[0], w) / np.polyval(den, w)
        return y

    def make_state(  # pylint: disable=arguments-renamed
        self, shape_in, shape_out, dt, dtype=None, y0=0
    ):
        assert shape_in == shape_out

        # call LinearSystem's `make_state`
        state = super().make_state(shape_in + (1,), shape_out + (1,), dt, dtype=dtype)
        X = state["X"]

        # initialize X using y0 as steady-state output
        y0 = np.array(y0, copy=False, ndmin=1)
        assert y0.shape[-1] == 1

        if (y0 == 0).all():
            # just leave X as zeros in this case, so that this value works
            # for unstable systems
            return state

        A, B, C, D = self.discrete_ss(dt)
        if LinearFilter.OneX.check(A, B, C, D, X):
            # OneX combines B and C into one scaling value `b`
            b = B.item() * C.item()
            X[:] = (b / (1 - A.item())) * y0
        else:
            # Solve for u0 (input) given y0 (output), then X given u0
            assert B.ndim == 1 or B.ndim == 2 and B.shape[1] == 1
            IAB = np.linalg.solve(np.eye(len(A)) - A, B)
            Q = C.dot(IAB) + D  # multiplier from input to output (DC gain)
            assert Q.size == 1
            if np.abs(Q.item()) > 1e-8:
                X[:] = y0.dot(IAB.T / Q.item())
            else:
                raise ValidationError(
                    "Cannot solve for state if DC gain is zero. Please set `y0=0`.",
                    "y0",
                    obj=self,
                )

        return state

    def make_step(self, shape_in, shape_out, dt, rng, state):
        """Returns a `.Step` instance that implements the linear filter."""
        assert shape_in == shape_out
        assert state is not None
        assert self.input_size == self.output_size == 1

        A, B, C, D = self.discrete_ss(dt)
        X = state["X"]

        if LinearFilter.NoX.check(A, B, C, D, X):
            return LinearFilter.NoX(A, B, C, D, X)
        if LinearFilter.OneXScalar.check(A, B, C, D, X):
            return LinearFilter.OneXScalar(A, B, C, D, X)
        elif LinearFilter.OneX.check(A, B, C, D, X):
            return LinearFilter.OneX(A, B, C, D, X)
        elif LinearFilter.NoD.check(A, B, C, D, X):
            return LinearFilter.NoD(A, B, C, D, X)
        else:
            assert LinearFilter.General.check(A, B, C, D, X)
            return LinearFilter.General(A, B, C, D, X)

    class Step:
        """Abstract base class for LTI filtering step functions."""

        def __init__(self, A, B, C, D, X):
            if not self.check(A, B, C, D, X):
                raise ValidationError(
                    "Matrices do not meet the requirements for this Step",
                    attr="A,B,C,D,X",
                    obj=self,
                )
            self.AT = A.T
            self.BT = B.T
            self.CT = C.T
            self.DT = D.T
            self.X = X

        def __call__(self, t, signal):
            raise NotImplementedError("Step object must implement __call__")

        @classmethod
        def check(cls, A, B, C, D, X):
            if A.size == 0 or B.size == 0:
                return X.size == A.size == B.size == C.size == 0 and D.size == 1
            else:
                return (
                    A.shape[0] == A.shape[1] == B.shape[0] == C.shape[1]
                    and A.shape[0] == X.shape[-1]
                    and C.shape[0] == B.shape[1] == 1
                    and D.size == 1
                )

    class NoX(Step):
        """Step for system with no state, only passthrough matrix (D)."""

        def __init__(self, A, B, C, D, X):
            super().__init__(A, B, C, D, X)
            self.d = D.item()

        def __call__(self, t, signal):
            return self.d * signal

        @classmethod
        def check(cls, A, B, C, D, X):
            return super().check(A, B, C, D, X) and _is_empty(A) and _is_empty(B)

    class OneX(Step):
        """Step for systems with one state element and no passthrough (D)."""

        def __init__(self, A, B, C, D, X):
            super().__init__(A, B, C, D, X)
            self.a = A.item()
            self.b = C.item() * B.item()

        def __call__(self, t, signal):
            self.X *= self.a
            self.X += self.b * signal[..., None]
            return self.X.squeeze(axis=-1)

        @classmethod
        def check(cls, A, B, C, D, X):
            return super().check(A, B, C, D, X) and len(A) == 1 and _is_empty(D)

    class OneXScalar(OneX):
        """Step for systems with one state element, no passthrough, and a size-1 input.

        Using the builtin float math improves performance.
        """

        def __call__(self, t, signal):
            self.X[:] = self.a * self.X.item() + self.b * signal.item()
            return self.X[0]

        @classmethod
        def check(cls, A, B, C, D, X):
            return super().check(A, B, C, D, X) and X.size == 1

    class NoD(Step):
        """Step for systems with no passthrough matrix (D).

        Implements::

            x[t] = A x[t-1] + B u[t]
            y[t] = C x[t]

        Note how the input has been advanced one step as compared with the
        General system below, to remove the unnecessary delay.
        """

        def __call__(self, t, signal):
            self.X[:] = np.dot(self.X, self.AT) + signal[..., None] * self.BT
            return np.dot(self.X, self.CT).squeeze(axis=-1)

        @classmethod
        def check(cls, A, B, C, D, X):
            return super().check(A, B, C, D, X) and len(A) >= 1 and _is_empty(D)

    class General(Step):
        """Step for any LTI system with at least one state element (X).

        Implements::

            x[t+1] = A x[t] + B u[t]
            y[t] = C x[t] + D u[t]

        Use ``NoX`` for systems with no state elements.
        """

        def __call__(self, t, signal):
            Y = np.dot(self.X, self.CT).squeeze(axis=-1) + signal * self.DT
            self.X[:] = np.dot(self.X, self.AT) + signal[..., None] * self.BT
            return Y

        @classmethod
        def check(cls, A, B, C, D, X):
            return super().check(A, B, C, D, X) and len(A) >= 1


class Lowpass(LinearFilter):
    r"""Standard first-order lowpass filter synapse.

    The impulse-response function (time domain) and transfer function are:

    .. math::

        h(t) &= (1 / \tau) \exp(-t / \tau) \\
        H(s) &= \frac{1}{\tau s + 1}

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.

    Attributes
    ----------
    tau : float
        The time constant of the filter in seconds.
    """

    tau = NumberParam("tau", low=0)

    def __init__(self, tau, **kwargs):
        self.tau = tau
        super().__init__(([1], [tau, 1]), analog=True, **kwargs)

    @property
    def cutoff(self):
        """Cutoff frequency in Hz; frequencies above this are attenuated."""
        return 1 / (2 * np.pi * self.tau)


class Alpha(LinearFilter):
    r"""Alpha-function filter synapse.

    The impulse-response function (time domain) and transfer function are:

    .. math::

        h(t) &= (t / \tau^2) \exp(-t / \tau) \\
        H(s) &= \frac{1}{(\tau s + 1)^2}

    This was found by [1]_ to be a good basic model for synapses.

    Parameters
    ----------
    tau : float
        The time constant of the filter in seconds.

    Attributes
    ----------
    tau : float
        The time constant of the filter in seconds.

    References
    ----------
    .. [1] Mainen, Z.F. and Sejnowski, T.J. (1995). Reliability of spike timing
       in neocortical neurons. Science (New York, NY), 268(5216):1503-6.
    """

    tau = NumberParam("tau", low=0)

    def __init__(self, tau, **kwargs):
        self.tau = tau
        super().__init__(([1], [tau ** 2, 2 * tau, 1]), analog=True, **kwargs)

    @property
    def cutoff(self):
        """Cutoff frequency in Hz; frequencies above this are attenuated."""
        return np.sqrt(np.sqrt(2) - 1) / (2 * np.pi * self.tau)


class DoubleExp(LinearFilter):
    r"""A second-order (two-pole) lowpass filter synapse with no zeros.

    The transfer function is:

    .. math::

        h(t) &= \frac{\exp(-t / \tau_1) - \exp(-t / \tau_2)}{\tau_1 - \tau_2} \\
        H(s) &= \frac{1}{(\tau_1 s + 1)(\tau_2 s + 1)}

    Equivalent to convolving two lowpass synapses together with potentially
    different time-constants, in either order.

    Parameters
    ----------
    tau1 : ``float``
        Time-constant of one exponential decay.
    tau2 : ``float``
        Time-constant of the other exponential decay.

    See Also
    --------
    :class:`.Lowpass`
    :class:`.Alpha`

    Examples
    --------
    .. testcode::

        sys = nengo.synapses.DoubleExp(1, 0.01)
        cutoffs = [sys.cutoff_low, sys.cutoff_high]
        freqs = np.logspace(-2, 2, 100)
        gain = np.abs(sys.evaluate(freqs))
        plt.semilogx(
            freqs,
            20 * np.log10(gain),
            label=r"$F_L = %0.3f$, $F_H = %0.3f$" % tuple(cutoffs),
        )
        cutoff_gains = np.abs(sys.evaluate(cutoffs))
        plt.semilogx(cutoffs, 20 * np.log10(cutoff_gains), "x", label="cutoffs")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.legend()
    """

    tau1 = NumberParam("tau1", low=0)
    tau2 = NumberParam("tau2", low=0)

    def __init__(self, tau1, tau2, **kwargs):
        self.tau1 = tau1
        self.tau2 = tau2
        super().__init__(([1], [tau1 * tau2, tau1 + tau2, 1]), analog=True, **kwargs)

    @property
    def cutoff_low(self):
        """Lower cutoff frequency in Hz; above this, attenuation is 20 dB/decade."""
        return 1 / (2 * np.pi * max(self.tau1, self.tau2))

    @property
    def cutoff_high(self):
        """Higher cutoff frequency in Hz; above this, attenuation is 40 dB/decade."""
        return 1 / (2 * np.pi * min(self.tau1, self.tau2))


class Bandpass(LinearFilter):
    r"""A second-order bandpass with given frequency and width.

    Implements the transfer function:

    .. math:: H(s) = \frac{\alpha w_0 s}{s^2 + \alpha w_0 s + w_0^2}

    where :math:`w_0 = 2 * \pi * freq` is the peak angular frequency, and
    :math:`\alpha` determines the width of the pass band. [1]_

    Given desired ``cutoff_low`` and ``cutoff_high``, the low and high frequencies at
    which the attenuation reaches -3 dB, respectively, ``freq`` and ``alpha`` are::

        freq = sqrt(cutoff_low * cutoff_high)
        alpha = (cutoff_high - cutoff_low) / freq

    Parameters
    ----------
    freq : ``float``
        Frequency (in hertz) of the peak of the pass band.
    alpha : ``float``
        Proportional to width of the pass band. Inverse of the Q factor of the system.

    References
    ----------
    .. [1] Hank Zumbahlen (Ed.), "Basic Linear Design", 2007. chapter 8.
       http://www.analog.com/library/analogDialogue/archives/43-09/EDCh%208%20filter.pdf

    Examples
    --------
    Plot frequency responses of bandpass filters centered around 2 Hz with varying
    pass-band widths:

    .. testcode::

        freqs = np.logspace(-2, 2, 100)
        for alpha in (0.1, 0.2, 0.5, 1, 2):
            sys = nengo.synapses.Bandpass(2, alpha)
            gain = np.abs(sys.evaluate(freqs))
            plt.semilogx(
                freqs,
                20 * np.log10(gain),
                label=r"$\alpha = %s$, $F_L = %0.3f$, $F_H = %0.3f$" % (
                    alpha, sys.cutoff_low, sys.cutoff_high
                )
            )
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.legend()
    """

    freq = NumberParam("freq", low=0, low_open=True)
    alpha = NumberParam("alpha", low=0, low_open=True)

    def __init__(self, freq, alpha=1, **kwargs):
        self.freq = freq
        self.alpha = alpha
        w0 = freq * (2 * np.pi)
        super().__init__(
            ([alpha * w0, 0], [1, alpha * w0, w0 ** 2]), analog=True, **kwargs
        )

    @property
    def cutoff_low(self):
        """Low cutoff frequency in Hz; frequencies below this are attenuated."""
        return (np.sqrt(1 + 0.25 * self.alpha ** 2) - 0.5 * self.alpha) * self.freq

    @property
    def cutoff_high(self):
        """High cutoff frequency in Hz; frequencies above this are attenuated."""
        return (np.sqrt(1 + 0.25 * self.alpha ** 2) + 0.5 * self.alpha) * self.freq


class Highpass(LinearFilter):
    r"""A highpass filter of given order.

    The transfer function is given by:

    .. math:: H(s) = \left( \frac{\tau s}{\tau s + 1} \right)^{order}

    Equivalent to differentiating the input, scaling by :math:`\tau`,
    lowpass filtering with time-constant :math:`\tau`, and finally repeating
    this ``order`` times. The lowpass filter is required to make this causal.

    The ``cutoff`` frequency is given by :math:`1 / (2 \pi \tau)`.

    Parameters
    ----------
    tau : ``float``
        Time-constant of the lowpass filter, and highpass gain.
    order : ``integer``, optional
        Dimension of the resulting linear system.

    See Also
    --------
    :class:`.Lowpass`

    Examples
    --------
    Evaluate the highpass in the frequency domain with a time-constant of 40 ms
    (a cutoff of about 4 Hz), and order 2:

    .. testcode::

        sys = nengo.synapses.Highpass(0.04, order=2)
        freqs = np.logspace(-2, 2, 100)
        gain = np.abs(sys.evaluate(freqs))
        plt.semilogx(freqs, 20 * np.log10(gain), label="cutoff = %0.3f" % (sys.cutoff))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.legend()
    """

    tau = NumberParam("tau", low=0)
    order = IntParam("order", low=1)

    def __init__(self, tau, order=1, **kwargs):
        self.tau = tau
        self.order = order
        num = (np.poly1d([tau, 0]) ** order).coeffs
        den = (np.poly1d([tau, 1]) ** order).coeffs
        super().__init__((num, den), analog=True, **kwargs)

    @property
    def cutoff(self):
        """Cutoff frequency in Hz; frequencies below this are attenuated."""
        return 1 / (2 * np.pi * self.tau)


class DiscreteDelay(LinearFilter):
    """A discrete (pure) time-delay of a given number of steps.

    Described by the discrete transfer function

    .. math:: H(z) = z^{-steps}

    Parameters
    ----------
    steps : ``integer``
        Number of time-steps to delay the input signal.

    See Also
    --------
    :class:`.LegendreDelay`

    Notes
    -----
    A single step of the delay will be removed if using the ``filt`` method.
    This is done for subtle reasons of consistency with `.Simulator`.
    The correct delay will appear when passed to `.Connection`.

    Examples
    --------
    Simulate a network using a discrete delay of 0.3 seconds for a synapse:

    .. testcode::

        from nengo.synapses import DiscreteDelay

        dt = 0.001
        with nengo.Network() as model:
            stim = nengo.Node(output=lambda t: np.sin(2*np.pi*t))
            p_stim = nengo.Probe(stim)
            p_delay = nengo.Probe(stim, synapse=DiscreteDelay(int(0.3 / dt)))

        with nengo.Simulator(model, dt=dt) as sim:
            sim.run(1.)

        plt.plot(sim.trange(), sim.data[p_stim], label="Stimulus")
        plt.plot(sim.trange(), sim.data[p_delay], label="Delayed")
        plt.xlabel("Time (s)")
        plt.legend()

    .. testoutput::
       :hide:

       ...
    """

    steps = IntParam("steps", low=0)

    def __init__(self, steps, **kwargs):
        self.steps = steps
        super().__init__(([1], [1] + [0] * steps), analog=False, **kwargs)


class LegendreDelay(LinearFilter):
    r"""A finite-order approximation of a pure delay, using the shifted Legendre basis.

    Implements the transfer function:

    .. math:: H(s) = e^{-\theta s} + \mathcal{O}(s^{2 n})

    where :math:`n` is the order of the system. This results in a pure delay of
    :math:`\theta` seconds (i.e. :math:`y(t) \approx u(t - \theta)` in the time-domain),
    for slowly changing inputs.

    Its canonical state-space realization represents the window of history
    by the shifted Legendre polnomials:

    .. math:: P_i(2 \theta' \theta^{-1} - 1)

    where ``i`` is the zero-based index into the state-vector.

    Parameters
    ----------
    theta : ``float``
        Length of time-delay in seconds.
    order : ``integer``
        Order of approximation in the denominator
        (dimensionality of resulting system).

    See Also
    --------
    :class:`.DiscreteDelay`

    Examples
    --------
    Delay 15 Hz band-limited white noise by 100 ms using various orders of
    approximations:

    .. testcode::

        process = nengo.processes.WhiteSignal(10.0, high=15, y0=0)
        u = process.run_steps(500)
        t = process.ntrange(len(u))

        plt.plot(t[100:], u[:-100], linestyle="--", lw=4, label="Ideal")
        for order in list(range(4, 8)):
            sys = nengo.synapses.LegendreDelay(0.1, order=order)
            plt.plot(t, sys.filt(u), label="order=%s" % order)

        plt.xlabel("Time (s)")
        plt.legend()
    """

    theta = NumberParam("theta", low=0)
    order = IntParam("order", low=1)

    def __init__(self, theta, order, **kwargs):
        self.theta = theta
        self.order = order

        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))

        super().__init__((A, B, C, D), analog=True, **kwargs)


class Triangle(Synapse):
    """Triangular finite impulse response (FIR) synapse.

    This synapse has a triangular and finite impulse response. The length of
    the triangle is ``t`` seconds; thus the digital filter will have
    ``t / dt + 1`` taps.

    Parameters
    ----------
    t : float
        Length of the triangle, in seconds.

    Attributes
    ----------
    t : float
        Length of the triangle, in seconds.
    """

    t = NumberParam("t", low=0)

    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def _get_coefficients(self, dt, dtype=None):
        dtype = rc.float_dtype if dtype is None else np.dtype(dtype)

        n_taps = int(np.round(self.t / float(dt))) + 1
        num = np.arange(n_taps, 0, -1, dtype=rc.float_dtype)
        num /= num.sum()

        # Minimal multiply implementation finds the difference between
        # coefficients and subtracts a scaled signal at each time step.
        n0, ndiff = num[0], num[-1]

        return n_taps, n0, ndiff

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=0):
        assert shape_in == shape_out
        dtype = rc.float_dtype if dtype is None else np.dtype(dtype)

        n_taps, _, ndiff = self._get_coefficients(dt, dtype=dtype)
        Y = np.zeros(shape_out, dtype=dtype)
        X = np.zeros((n_taps,) + shape_out, dtype=dtype)
        Xi = np.zeros(1, dtype=dtype)  # counter for X position

        if y0 != 0 and len(X) > 0:
            y0 = np.array(y0, copy=False, ndmin=1)
            X[:] = ndiff * y0[None, ...]
            Y[:] = y0

        return {"Y": Y, "X": X, "Xi": Xi}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == shape_out
        assert state is not None

        Y, X, Xi = state["Y"], state["X"], state["Xi"]
        n_taps, n0, ndiff = self._get_coefficients(dt, dtype=Y.dtype)
        assert len(X) == n_taps

        def step_triangle(t, signal):
            Y[...] += n0 * signal
            Y[...] -= X.sum(axis=0)
            Xi[:] = (Xi + 1) % len(X)
            X[int(Xi.item())] = ndiff * signal
            return Y

        return step_triangle


class SynapseParam(Parameter):
    equatable = True

    def __init__(self, name, default=Unconfigurable, optional=True, readonly=None):
        super().__init__(name, default, optional, readonly)

    def coerce(self, instance, synapse):  # pylint: disable=arguments-renamed
        synapse = Lowpass(synapse) if is_number(synapse) else synapse
        self.check_type(instance, synapse, Synapse)
        return super().coerce(instance, synapse)


class LinearFilterParam(SynapseParam):
    equatable = True

    def coerce(self, instance, synapse):
        synapse = super().coerce(instance, synapse)
        self.check_type(instance, synapse, LinearFilter)
        return synapse
