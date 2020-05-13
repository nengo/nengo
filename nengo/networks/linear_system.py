import numpy as np

import nengo
from nengo.linear_system import LinearSystem, LinearSystemParam
from nengo.synapses import LinearSystem, LinearFilterParam


def ss2sim(sys, synapse, dt=None):
    """Maps a linear system onto a synapse in state-space form.

    This implements a generalization of Principle 3 from the Neural Engineering
    Framework (NEF). [#]_ Intuitively, this routine compensates for the change in
    dynamics that occurs when the integrator that usually forms the basis for any
    linear system is replaced by the given synapse.
    This is needed because in neural systems we don't have access to a
    perfect integrator; instead the synapse model becomes the
    "dynamical primitive".

    Parameters
    ----------
    sys : :data:`linear_system_like`
        Linear system representation of desired dynamical system.
        Requires ``sys.analog == synapse.analog``.
    synapse : :data:`linear_system_like`
        Linear system representation of the synapse providing the dynamics.
        Requires ``sys.analog == synapse.analog``.
    dt : ``float`` or ``None``
        Time-step of simulation. If not ``None``, then both ``sys`` and
        ``synapse`` are discretized using the ``'zoh'`` method.
        In either case, if ``sys`` is now digital, then the digital
        generalization of Principle 3 will be applied --- otherwise the analog
        version will be applied.

    Returns
    -------
    mapped_sys : :class:`.LinearSystem`
        Linear system whose state-space matrices yield the desired
        dynamics when using the synapse model instead of an integrator.

    See Also
    --------
    :class:`.LinearSystemNetwork`

    Notes
    -----
    This routine is called automatically by :class:`.LinearSystemNetwork`.

    Principle 3 is a special case of this routine when called with
    a continuous :func:`Lowpass` synapse and ``dt=None``. However, specifying
    the ``dt`` (or providing digital systems) will improve the accuracy in
    digital simulation.

    For higher-order synapses, this makes a zero-order hold (ZOH) assumption
    to avoid requiring the input derivatives. In this case, the mapping is
    not perfect. If the input derivatives are known, then the accuracy can be
    made perfect again. See references for details.

    References
    ----------
    .. [#] A. R. Voelker and C. Eliasmith, "Improving spiking dynamical
       networks: Accurate delays, higher-order synapses, and time cells",
       2017, Submitted. [`URL <https://github.com/arvoelke/delay2017>`__]

    Examples
    --------
    See :doc:`notebooks/research/discrete_comparison` for a notebook example.

    >>> from nengolib.synapses import ss2sim, PadeDelay

    Map the state of a balanced :func:`PadeDelay` onto a lowpass synapse:

    >>> import nengo
    >>> from nengolib.signal import balance
    >>> sys = balance(PadeDelay(.05, order=6))
    >>> synapse = nengo.Lowpass(.1)
    >>> mapped = ss2sim(sys, synapse, synapse.default_dt)
    >>> assert np.allclose(sys.C, mapped.C)
    >>> assert np.allclose(sys.D, mapped.D)

    Simulate the mapped system directly (without neurons):

    >>> process = nengo.processes.WhiteSignal(1, high=10, y0=0)
    >>> with nengo.Network() as model:
    >>>     stim = nengo.Node(output=process)
    >>>     x = nengo.Node(size_in=len(sys))
    >>>     nengo.Connection(stim, x, transform=mapped.B, synapse=synapse)
    >>>     nengo.Connection(x, x, transform=mapped.A, synapse=synapse)
    >>>     p_stim = nengo.Probe(stim)
    >>>     p_actual = nengo.Probe(x)
    >>> with nengo.Simulator(model) as sim:
    >>>     sim.run(.5)

    The desired dynamics are implemented perfectly:

    >>> target = sys.X.filt(sim.data[p_stim])
    >>> assert np.allclose(target, sim.data[p_actual])

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sim.trange(), target, linestyle='--', lw=4)
    >>> plt.plot(sim.trange(), sim.data[p_actual], alpha=.5)
    >>> plt.show()
    """

    if not isinstance(sys, LinearSystem):
        synapse = LinearSystem(sys)

    if not isinstance(synapse, LinearSystem):
        synapse = LinearSystem(synapse)

    if dt is not None and sys.analog:
        sys = sys.discretize(dt=dt)

    if dt is not None and synapse.analog:
        synapse = synapse.discretize(dt=dt)

    if synapse.analog != sys.analog:
        raise ValueError("If `dt is None`, `sys.analog` must equal `synapse.analog`")

    num, den = synapse.tf
    if num.ndim == 2:
        if num.shape[0] == 1:
            num = num[0]
        else:
            raise ValueError("Synapse must be a single-output system")

    assert num.ndim == 1

    num_max = np.max(num)
    num[np.abs(num) < 1e-8 * num_max] = 0

    num, den = np.poly1d(num), np.poly1d(den)
    if synapse.analog and len(num.coeffs) > 1:
        raise ValueError("analog synapses (%s) must not have zeros" % synapse)

    # If the synapse was discretized, then its numerator may now have multiple
    #   coefficients. By summing them together, we are implicitly assuming that
    #   the output of the synapse will stay constant across
    #   synapse.order_num + 1 time-steps. This is also related to:
    #   http://dsp.stackexchange.com/questions/33510/difference-between-convolving-before-after-discretizing-lti-systems  # noqa: E501
    # For example, if we have H = Lowpass(0.1), then the only difference
    #   between sys1 = cont2discrete(H*H, dt) and
    #           sys2 = cont2discrete(H, dt)*cont2discrete(H, dt), is that
    #   np.sum(sys1.num) == sys2.num (while sys1.den == sys2.den)!
    gain = np.sum(num)
    c = den / gain

    A, B, C, D = sys.ss
    k = synapse.state_size
    powA = [np.linalg.matrix_power(A, i) for i in range(k + 1)]

    AH = np.sum([c[i] * powA[i] for i in range(k + 1)], axis=0)
    BH = np.dot(
        sum(c[i] * powA[i - j - 1] for j in range(k) for i in range(j + 1, k + 1)), B,
    )

    return LinearSystem((AH, BH, C, D), analog=sys.analog, x0=sys.x0)


class LinearSystemNetwork(nengo.Network):
    """Implement an arbitrary linear system in neurons.

    Takes a linear system and uses `.ss2sim` to account for the effect of the
    input/recurrent synapse on the system dynamics. The resulting network will provide
    an accurate implementation of the system using the provided synapse.

    Parameters
    ----------
    system : `.LinearSystem`
        The LinearSystem to be implemented.
    synapse : `.LinearFilter`
        The synapse to use on the recurrent connection and input connection. Must be
        an instance or subclass of `.LinearFilter`.
    n_neurons : int
        The number of neurons in the state (``X``) ensemble.
    dt : float (optional)
        The time constant used to discretize the synapses. If provided, it should be
        the same as the simulator time constant. If not provided (default), the system
        will not account for the effects of discretization.

    Attributes
    ----------
    mapped_system : `.LinearSystem`
        The mapped linear system that will be implemented by .
    It accounts for the effect of the recurrent/input synapses on the dynamics.
    input : `.Node`
        A passthrough `.Node` for connecting the system input.
    X : `.Ensemble`
        The `.Ensemble` representing the system state.
    output : `.Node`
        A passthrough `.Node` for retreiving the system output.
    ss_connections : dict of (str, `.Connection`)
        Dictionary containing a `.Connection` for each of the state-space transform
        matrices "A", "B", "C", and "D". "B" will not be present if the system does
        not have input, and "D" will not be present if the ``D`` matrix is zero.
    """

    system = LinearSystemParam("system")
    synapse = LinearFilterParam("synapse")

    def __init__(
        self,
        system,
        synapse,
        n_neurons=100,
        dt=None,
        output_synapse=None,
        label=None,
        seed=None,
        add_to_container=None,
        **ens_kwargs
    ):
        super().__init__(label, seed, add_to_container)

        self.system = system
        self.synapse = synapse
        self.mapped_system = ss2sim(self.system, self.synapse, dt=dt)
        A, B, C, D = self.mapped_system.ss

        self.ss_connections = {}
        self.X = nengo.Ensemble(
            n_neurons, dimensions=self.mapped_system.state_size, **ens_kwargs
        )
        self.ss_connections["A"] = nengo.Connection(
            self.X,
            self.X,
            transform=A,
            synapse=self.synapse,
            initial_value=self.mapped_system.x0,
        )

        if system.input_size == 0:
            self.input = None
        else:
            self.input = nengo.Node(size_in=system.input_size)
            self.ss_connections["B"] = nengo.Connection(
                self.input, self.X, transform=B, synapse=self.synapse
            )

        self.output = nengo.Node(size_in=system.output_size)
        self.ss_connections["C"] = nengo.Connection(
            self.X, self.output, transform=C, synapse=output_synapse,
        )

        if D.size > 0 and (D != 0).any():
            self.ss_connections["D"] = nengo.Connection(
                self.input, self.output, transform=D, synapse=output_synapse,
            )
