from fnmatch import fnmatch
from inspect import getfullargspec

import numpy as np
import pytest

import nengo
from nengo import (
    BCM,
    LIF,
    PES,
    RLS,
    AdaptiveLIF,
    AdaptiveLIFRate,
    Alpha,
    Convolution,
    Dense,
    Direct,
    Izhikevich,
    LIFRate,
    LinearFilter,
    Lowpass,
    Oja,
    RectifiedLinear,
    Sigmoid,
    Sparse,
    SpikingRectifiedLinear,
    Tanh,
    Voja,
)
from nengo.builder.learning_rules import SimBCM, SimOja, SimPES, SimRLS, SimVoja
from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import (
    Copy,
    DotInc,
    ElementwiseInc,
    Reset,
    SimPyFunc,
    TimeUpdate,
)
from nengo.builder.processes import SimProcess
from nengo.builder.signal import Signal
from nengo.dists import (
    PDF,
    Choice,
    CosineSimilarity,
    Exponential,
    Gaussian,
    Samples,
    ScatteredHypersphere,
    SqrtBeta,
    SubvectorLength,
    Uniform,
    UniformHypersphere,
)
from nengo.processes import (
    BrownNoise,
    FilteredNoise,
    Piecewise,
    PresentInput,
    WhiteNoise,
    WhiteSignal,
)
from nengo.solvers import (
    Lstsq,
    LstsqL1,
    LstsqL2,
    LstsqL2nz,
    LstsqNoise,
    Nnls,
    NnlsL2,
    NnlsL2nz,
    NoSolver,
)
from nengo.synapses import Triangle
from nengo.transforms import ChannelShape, NoTransform, SparseMatrix


def check_init_args(cls, args):
    assert getfullargspec(cls.__init__).args[1:] == args


def check_repr(obj):
    """Tests that repr gives back what is needed to create the object."""
    assert not isinstance(obj, str), "Passing a string doesn't test `repr(obj)`"

    # some reprs need these in the local namespace
    array = np.array
    int64 = np.int64
    assert array
    assert int64
    assert eval(repr(obj)) == obj  # pylint: disable = eval-used


def test_core_objects():
    with nengo.Network() as net1:

        a = nengo.Ensemble(100, 2)
        with nengo.Network(label="net2") as net2:
            b = nengo.Ensemble(100, 2, label="b")

        ap = nengo.Probe(a)
        bp = nengo.Probe(b)

        c1 = nengo.Connection(a, b)
        c2 = nengo.Connection(a, b, function=np.square)
        c3 = nengo.Connection(a, b, learning_rule_type=nengo.PES(), label="c3")

        n1 = nengo.Node(output=np.sin)
        n2 = nengo.Node(output=np.cos, label="n2")

    assert fnmatch(repr(net1), "<Network (unlabeled) at 0x*>")
    assert fnmatch(repr(net2), '<Network "net2" at 0x*>')

    # Ensemble
    assert fnmatch(repr(a), "<Ensemble (unlabeled) at 0x*>")
    assert fnmatch(repr(b), "<Ensemble 'b' at 0x*>")

    # Probe
    assert fnmatch(
        repr(ap), "<Probe at 0x* of 'decoded_output' of <Ensemble (unlabeled) at 0x*>>"
    )
    assert fnmatch(repr(bp), "<Probe at 0x* of 'decoded_output' of <Ensemble 'b'>>")

    # Connection
    assert fnmatch(
        repr(c1),
        "<Connection at 0x* from <Ensemble (unlabeled) at 0x*> to <Ensemble 'b'>>",
    )
    assert fnmatch(
        repr(c2),
        "<Connection at 0x* from <Ensemble (unlabeled) at 0x*> to <Ensemble 'b'> "
        "computing 'square'>",
    )
    assert fnmatch(repr(c3), "<Connection at 0x* c3>")

    assert fnmatch(
        repr(c3.learning_rule),
        "<LearningRule at 0x* modifying <Connection at 0x* c3> with type PES()>",
    )

    # Node
    assert fnmatch(repr(n1), "<Node (unlabeled) at 0x*>")
    assert fnmatch(repr(n2), "<Node 'n2' at 0x*>")

    # Neurons
    assert fnmatch(repr(b.neurons), "<Neurons at 0x* of <Ensemble 'b' at 0x*>>")

    # ObjView
    assert fnmatch(repr(a[:1]), "<Ensemble (unlabeled) at 0x*>[[]:1[]]")


def test_neuron_types():
    check_init_args(Direct, ["initial_state"])
    check_repr(Direct())
    assert repr(Direct()) == "Direct()"

    check_init_args(RectifiedLinear, ["amplitude", "initial_state"])
    check_repr(RectifiedLinear())
    check_repr(RectifiedLinear(amplitude=2))
    assert repr(RectifiedLinear()) == "RectifiedLinear()"

    check_init_args(SpikingRectifiedLinear, ["amplitude", "initial_state"])
    check_repr(SpikingRectifiedLinear())
    check_repr(SpikingRectifiedLinear(amplitude=2))
    check_repr(SpikingRectifiedLinear(initial_state={"voltage": Choice([1.0])}))
    assert repr(SpikingRectifiedLinear()) == "SpikingRectifiedLinear()"

    check_init_args(Sigmoid, ["tau_ref", "initial_state"])
    check_repr(Sigmoid())
    check_repr(Sigmoid(tau_ref=0.1))
    assert repr(Sigmoid()), "Sigmoid()"
    assert repr(Sigmoid(tau_ref=0.001)) == "Sigmoid(tau_ref=0.001)"

    check_init_args(LIFRate, ["tau_rc", "tau_ref", "amplitude", "initial_state"])
    check_repr(LIFRate())
    check_repr(LIFRate(tau_rc=0.1))
    check_repr(LIFRate(tau_ref=0.1))
    check_repr(LIFRate(amplitude=2))
    check_repr(LIFRate(tau_rc=0.05, tau_ref=0.02))
    check_repr(LIFRate(tau_rc=0.05, amplitude=2))
    check_repr(LIFRate(tau_ref=0.02, amplitude=2))
    check_repr(LIFRate(tau_rc=0.05, tau_ref=0.02, amplitude=2))
    assert repr(LIFRate()) == "LIFRate()"
    assert repr(LIFRate(tau_rc=0.01, tau_ref=0)) == "LIFRate(tau_rc=0.01, tau_ref=0)"

    check_init_args(
        LIF, ["tau_rc", "tau_ref", "min_voltage", "amplitude", "initial_state"]
    )
    check_repr(LIF())
    check_repr(LIF(tau_rc=0.1))
    check_repr(LIF(tau_ref=0.1))
    check_repr(LIF(amplitude=2))
    check_repr(LIF(min_voltage=-0.5))
    check_repr(LIF(tau_rc=0.05, tau_ref=0.02))
    check_repr(LIF(tau_rc=0.05, amplitude=2))
    check_repr(LIF(tau_ref=0.02, amplitude=2))
    check_repr(LIF(tau_rc=0.05, tau_ref=0.02, amplitude=2))
    check_repr(LIF(tau_rc=0.05, tau_ref=0.02, min_voltage=-0.5, amplitude=2))
    check_repr(LIF(initial_state={"refractory_time": Choice([0.1])}))
    assert repr(LIF()) == "LIF()"
    assert repr(LIF(tau_rc=0.01, tau_ref=0)) == "LIF(tau_rc=0.01, tau_ref=0)"

    check_init_args(
        AdaptiveLIFRate,
        ["tau_n", "inc_n", "tau_rc", "tau_ref", "amplitude", "initial_state"],
    )
    check_repr(AdaptiveLIFRate())
    check_repr(AdaptiveLIFRate(tau_n=0.1))
    check_repr(AdaptiveLIFRate(inc_n=0.5))
    check_repr(AdaptiveLIFRate(tau_rc=0.1))
    check_repr(AdaptiveLIFRate(tau_ref=0.1))
    check_repr(AdaptiveLIFRate(amplitude=2))
    check_repr(
        AdaptiveLIFRate(tau_n=0.1, inc_n=0.5, tau_rc=0.05, tau_ref=0.02, amplitude=2)
    )
    check_repr(AdaptiveLIFRate(initial_state={"adaptation": Choice([0.1])}))
    assert repr(AdaptiveLIFRate()) == "AdaptiveLIFRate()"
    assert (
        repr(AdaptiveLIFRate(tau_rc=0.01, tau_n=0.5, inc_n=0.02))
        == "AdaptiveLIFRate(tau_n=0.5, inc_n=0.02, tau_rc=0.01)"
    )

    check_init_args(
        AdaptiveLIF,
        [
            "tau_n",
            "inc_n",
            "tau_rc",
            "tau_ref",
            "min_voltage",
            "amplitude",
            "initial_state",
        ],
    )
    check_repr(AdaptiveLIF())
    check_repr(AdaptiveLIF(tau_n=0.1))
    check_repr(AdaptiveLIF(inc_n=0.5))
    check_repr(AdaptiveLIF(tau_rc=0.1))
    check_repr(AdaptiveLIF(tau_ref=0.1))
    check_repr(AdaptiveLIF(min_voltage=-0.5))
    check_repr(
        AdaptiveLIF(
            tau_n=0.1,
            inc_n=0.5,
            tau_rc=0.05,
            tau_ref=0.02,
            min_voltage=-0.5,
            amplitude=2,
        )
    )
    check_repr(AdaptiveLIF(initial_state={"adaptation": Choice([0.1])}))
    assert repr(AdaptiveLIF()) == "AdaptiveLIF()"
    assert (
        repr(AdaptiveLIF(tau_rc=0.01, tau_n=0.5, inc_n=0.02))
        == "AdaptiveLIF(tau_n=0.5, inc_n=0.02, tau_rc=0.01)"
    )

    check_init_args(
        Izhikevich,
        [
            "tau_recovery",
            "coupling",
            "reset_voltage",
            "reset_recovery",
            "initial_state",
        ],
    )
    check_repr(Izhikevich())
    check_repr(Izhikevich(tau_recovery=0.1))
    check_repr(Izhikevich(coupling=0.3))
    check_repr(Izhikevich(reset_voltage=-1))
    check_repr(Izhikevich(reset_recovery=5))
    check_repr(
        Izhikevich(tau_recovery=0.1, coupling=0.3, reset_voltage=-1, reset_recovery=5)
    )
    check_repr(Izhikevich(initial_state={"recovery": Choice([0.1])}))
    assert repr(Izhikevich()) == "Izhikevich()"
    assert (
        repr(
            Izhikevich(
                tau_recovery=0.01, coupling=0.5, reset_voltage=-60, reset_recovery=6
            )
        )
        == "Izhikevich(tau_recovery=0.01, coupling=0.5, reset_voltage=-60, "
        "reset_recovery=6)"
    )

    check_init_args(Tanh, ["tau_ref", "initial_state"])
    check_repr(Tanh())
    check_repr(Tanh(tau_ref=0.1))


def test_learning_rule_types():
    check_init_args(PES, ["learning_rate", "pre_synapse"])
    check_repr(PES(learning_rate=0.1, pre_synapse=Lowpass(tau=0.2)))
    assert repr(PES()) == "PES()"
    assert (
        repr(PES(learning_rate=0.1, pre_synapse=0.2))
        == "PES(learning_rate=0.1, pre_synapse=Lowpass(tau=0.2))"
    )

    check_init_args(
        BCM, ["learning_rate", "pre_synapse", "post_synapse", "theta_synapse"]
    )
    check_repr(
        BCM(learning_rate=0.1, pre_synapse=0.2, post_synapse=0.3, theta_synapse=0.4)
    )
    assert repr(BCM()) == "BCM()"
    assert repr(
        BCM(learning_rate=0.1, pre_synapse=0.2, post_synapse=0.3, theta_synapse=0.4)
    ) == (
        "BCM(learning_rate=0.1, pre_synapse=Lowpass(tau=0.2), "
        "post_synapse=Lowpass(tau=0.3), theta_synapse=Lowpass(tau=0.4))"
    )

    check_init_args(Oja, ["learning_rate", "pre_synapse", "post_synapse", "beta"])
    check_repr(
        Oja(
            learning_rate=0.1,
            pre_synapse=Lowpass(tau=0.2),
            post_synapse=Lowpass(tau=0.3),
            beta=0.4,
        )
    )
    assert repr(Oja()) == "Oja()"
    assert repr(
        Oja(learning_rate=0.1, pre_synapse=0.2, post_synapse=0.3, beta=0.4)
    ) == (
        "Oja(learning_rate=0.1, pre_synapse=Lowpass(tau=0.2), "
        "post_synapse=Lowpass(tau=0.3), beta=0.4)"
    )

    check_init_args(Voja, ["learning_rate", "post_synapse"])
    check_repr(Voja(learning_rate=0.1, post_synapse=Lowpass(tau=0.2)))
    assert repr(Voja()) == "Voja()"
    assert (
        repr(Voja(learning_rate=0.1, post_synapse=0.2))
        == "Voja(learning_rate=0.1, post_synapse=Lowpass(tau=0.2))"
    )

    check_init_args(RLS, ["learning_rate", "pre_synapse"])
    check_repr(RLS(2.4e-3))
    check_repr(RLS(learning_rate=0.1, pre_synapse=Alpha(tau=0.2)))
    assert repr(RLS()) == "RLS()"
    assert (
        repr(RLS(learning_rate=0.1, pre_synapse=0.2))
        == "RLS(learning_rate=0.1, pre_synapse=Lowpass(tau=0.2))"
    )


def test_distributions():
    check_init_args(PDF, ["x", "p"])
    check_repr(PDF([1, 2, 3], [0.1, 0.8, 0.1]))
    assert (
        repr(PDF([1, 2], [0.4, 0.6])) == "PDF(x=array([1., 2.]), p=array([0.4, 0.6]))"
    )

    check_init_args(Uniform, ["low", "high", "integer"])
    check_repr(Uniform(1, 3))
    check_repr(Uniform(1, 4, integer=True))
    assert repr(Uniform(0, 1)) == "Uniform(low=0, high=1)"
    assert repr(Uniform(0, 5, integer=True)) == "Uniform(low=0, high=5, integer=True)"

    check_init_args(Gaussian, ["mean", "std"])
    check_repr(Gaussian(0, 2))
    assert repr(Gaussian(1, 0.1)) == "Gaussian(mean=1, std=0.1)"

    check_init_args(Exponential, ["scale", "shift", "high"])
    check_repr(Exponential(2.0))
    check_repr(Exponential(2.0, shift=0.1))
    check_repr(Exponential(2.0, shift=0.1, high=10.0))
    assert repr(Exponential(2.0)) == "Exponential(scale=2.0)"

    check_init_args(UniformHypersphere, ["surface", "min_magnitude"])
    check_repr(UniformHypersphere())
    check_repr(UniformHypersphere(surface=True))
    check_repr(UniformHypersphere(min_magnitude=0.3))
    assert repr(UniformHypersphere()) == "UniformHypersphere()"
    assert repr(UniformHypersphere(surface=True)) == "UniformHypersphere(surface=True)"

    check_init_args(
        ScatteredHypersphere, ["surface", "min_magnitude", "base", "method"]
    )
    check_repr(ScatteredHypersphere())
    check_repr(ScatteredHypersphere(surface=True))
    check_repr(ScatteredHypersphere(min_magnitude=0.3))
    check_repr(ScatteredHypersphere(base=Uniform(0, 1)))
    check_repr(ScatteredHypersphere(method="tfww"))
    assert repr(ScatteredHypersphere()) == "ScatteredHypersphere()"
    assert (
        repr(ScatteredHypersphere(surface=True)) == "ScatteredHypersphere(surface=True)"
    )
    assert (
        repr(ScatteredHypersphere(base=Uniform(0, 1), method="tfww"))
        == "ScatteredHypersphere(base=Uniform(low=0, high=1), method='tfww')"
    )

    check_init_args(Choice, ["options", "weights"])
    check_repr(Choice([3, 2, 1]))
    check_repr(Choice([3, 2, 1], weights=[0.1, 0.2, 0.7]))
    assert repr(Choice([1, 2, 3])) == "Choice(options=array([1., 2., 3.]))"
    assert (
        repr(Choice([1, 2, 3], weights=[0.1, 0.5, 0.4]))
        == "Choice(options=array([1., 2., 3.]), weights=array([0.1, 0.5, 0.4]))"
    )

    check_init_args(Samples, ["samples"])
    check_repr(Samples([3, 2, 1]))
    assert repr(Samples([3, 2, 1])) == "Samples(samples=array([3., 2., 1.]))"

    check_init_args(SqrtBeta, ["n", "m"])
    check_repr(SqrtBeta(3))
    check_repr(SqrtBeta(3, m=2))
    assert repr(SqrtBeta(3)) == "SqrtBeta(n=3)"
    assert repr(SqrtBeta(3, 2)) == "SqrtBeta(n=3, m=2)"

    check_init_args(SubvectorLength, ["dimensions", "subdimensions"])
    check_repr(SubvectorLength(6))
    check_repr(SubvectorLength(6, 2))
    assert repr(SubvectorLength(3)) == "SubvectorLength(dimensions=3)"

    check_init_args(CosineSimilarity, ["dimensions"])
    check_repr(CosineSimilarity(6))
    assert repr(CosineSimilarity(6)) == "CosineSimilarity(dimensions=6)"


def test_synapses():
    check_init_args(LinearFilter, ["num", "den", "analog", "method"])
    check_repr(LinearFilter([1, 2], [3, 4]))
    check_repr(LinearFilter([1, 2], [3, 4], analog=False))
    assert (
        repr(LinearFilter([1], [0.03, 1]))
        == "LinearFilter(num=array([1.]), den=array([0.03, 1.  ]))"
    )

    check_init_args(Lowpass, ["tau"])
    check_repr(Lowpass(0.3))
    assert repr(Lowpass(0.01)) == "Lowpass(tau=0.01)"

    check_init_args(Alpha, ["tau"])
    check_repr(Alpha(0.3))
    assert repr(Alpha(0.02)) == "Alpha(tau=0.02)"

    check_init_args(Triangle, ["t"])
    check_repr(Triangle(0.3))
    assert repr(Triangle(0.03)) == "Triangle(t=0.03)"


def test_processes():
    assert (
        repr(WhiteSignal(0.2, 10, rms=0.3))
        == "WhiteSignal(period=0.2, high=10, rms=0.3)"
    )

    check_init_args(WhiteNoise, ["dist", "scale"])
    check_repr(WhiteNoise(scale=False))
    assert repr(WhiteNoise()) == "WhiteNoise()"
    assert repr(WhiteNoise(scale=False)) == "WhiteNoise(scale=False)"

    check_init_args(FilteredNoise, ["synapse", "dist", "scale"])
    check_repr(FilteredNoise(scale=False))
    assert repr(FilteredNoise()) == "FilteredNoise()"
    assert repr(FilteredNoise(scale=False)) == "FilteredNoise(scale=False)"

    check_init_args(BrownNoise, ["dist"])
    check_repr(BrownNoise())
    assert repr(BrownNoise()) == "BrownNoise()"

    check_init_args(PresentInput, ["inputs", "presentation_time"])
    check_repr(PresentInput(inputs=np.array([1.2, 3.4]), presentation_time=5))
    assert (
        repr(PresentInput((1.2, 3.4), 5))
        == "PresentInput(inputs=array([1.2, 3.4]), presentation_time=5)"
    )

    check_init_args(WhiteSignal, ["period", "high", "rms", "y0"])
    check_repr(WhiteSignal(period=1.2, high=3.4, rms=5.6, y0=7.8))
    assert repr(WhiteSignal(1, 2)) == "WhiteSignal(period=1, high=2)"
    assert (
        repr(WhiteSignal(1.2, 3.4, 5.6, 7.8))
        == "WhiteSignal(period=1.2, high=3.4, rms=5.6, y0=7.8)"
    )

    check_init_args(Piecewise, ["data", "interpolation"])
    check_repr(Piecewise(data={1: 0.1, 2: 0.2, 3: 0.3}))
    assert (
        repr(Piecewise({1: 0.1, 2: 0.2, 3: 0.3}))
        == "Piecewise(data={1: array([0.1]), 2: array([0.2]), 3: array([0.3])})"
    )


def test_piecewise():
    pytest.importorskip("scipy.optimize")
    for interpolation in ("linear", "nearest", "slinear", "quadratic", "cubic"):
        assert repr(Piecewise({1: 0.1, 2: 0.2, 3: 0.3}, interpolation)) == (
            "Piecewise(data={1: array([0.1]), 2: array([0.2]), 3: array([0.3])}, "
            f"interpolation='{interpolation}')"
        )


def test_solvers():
    check_init_args(Lstsq, ["weights", "rcond"])
    check_repr(Lstsq(weights=True, rcond=0.1))
    assert repr(Lstsq(weights=True, rcond=0.1)) == "Lstsq(weights=True, rcond=0.1)"

    check_init_args(LstsqNoise, ["weights", "noise", "solver"])
    check_repr(LstsqNoise(weights=True, noise=0.2))
    assert (
        repr(LstsqNoise(weights=True, noise=0.2))
        == "LstsqNoise(weights=True, noise=0.2)"
    )

    check_init_args(LstsqL2, ["weights", "reg", "solver"])
    check_repr(LstsqL2(weights=True, reg=0.2))
    assert repr(LstsqL2(weights=True, reg=0.2)) == "LstsqL2(weights=True, reg=0.2)"

    check_init_args(LstsqL2nz, ["weights", "reg", "solver"])
    check_repr(LstsqL2nz(weights=True, reg=0.2))
    assert repr(LstsqL2nz(weights=True, reg=0.2)) == "LstsqL2nz(weights=True, reg=0.2)"

    check_init_args(NoSolver, ["values", "weights"])
    check_repr(NoSolver(values=np.array([[1.2, 3.4, 5.6, 7.8]]), weights=True))
    assert (
        repr(NoSolver([[1.2, 3.4, 5.6, 7.8]], weights=True))
        == "NoSolver(values=array([[1.2, 3.4, 5.6, 7.8]]), weights=True)"
    )


def test_lstsql1_repr():
    pytest.importorskip("sklearn")

    check_init_args(LstsqL1, ["weights", "l1", "l2", "max_iter"])
    check_repr(LstsqL1(weights=True, l1=0.2, l2=0.3, max_iter=4))
    assert (
        repr(LstsqL1(weights=True, l1=0.2, l2=0.3, max_iter=4))
        == "LstsqL1(weights=True, l1=0.2, l2=0.3, max_iter=4)"
    )


def test_nnls_repr():
    pytest.importorskip("scipy.optimize")

    check_init_args(Nnls, ["weights"])
    check_repr(Nnls(weights=True))
    assert repr(Nnls(weights=True)) == "Nnls(weights=True)"

    check_init_args(NnlsL2, ["weights", "reg"])
    check_repr(NnlsL2(weights=True, reg=0.2))
    assert repr(NnlsL2(weights=True, reg=0.2)) == "NnlsL2(weights=True, reg=0.2)"

    check_init_args(NnlsL2nz, ["weights", "reg"])
    check_repr(NnlsL2nz(weights=True, reg=0.2))
    assert repr(NnlsL2nz(weights=True, reg=0.2)) == "NnlsL2nz(weights=True, reg=0.2)"


def test_transforms():
    check_init_args(Dense, ["shape", "init"])
    # No check_repr because dense matrices are usually too big
    assert repr(Dense((1, 2), init=[[1, 1]])) == "Dense(shape=(1, 2))"

    check_init_args(
        Convolution,
        [
            "n_filters",
            "input_shape",
            "kernel_size",
            "strides",
            "padding",
            "channels_last",
            "init",
        ],
    )
    check_repr(Convolution(n_filters=3, input_shape=(1, 2, 3)))
    check_repr(Convolution(n_filters=3, input_shape=(1, 2, 3), kernel_size=(3, 2)))
    check_repr(Convolution(n_filters=3, input_shape=(1, 2, 3), channels_last=False))
    assert (
        repr(Convolution(n_filters=3, input_shape=(1, 2, 3)))
        == "Convolution(n_filters=3, input_shape=(1, 2, 3))"
    )
    assert (
        repr(Convolution(n_filters=3, input_shape=(1, 2, 3), kernel_size=(3, 2)))
        == "Convolution(n_filters=3, input_shape=(1, 2, 3), kernel_size=(3, 2))"
    )
    assert (
        repr(Convolution(n_filters=3, input_shape=(1, 2, 3), channels_last=False))
        == "Convolution(n_filters=3, input_shape=(1, 2, 3), channels_last=False)"
    )

    check_init_args(Sparse, ["shape", "indices", "init"])
    # No check_repr because sparse matrices are usually too big
    assert repr(Sparse((1, 1), indices=[[1, 1], [1, 1]])) == "Sparse(shape=(1, 1))"
    assert (
        repr(Sparse((1, 1), indices=[[1, 1], [1, 1], [1, 1]], init=2))
        == "Sparse(shape=(1, 1))"
    )

    check_init_args(SparseMatrix, ["indices", "data", "shape"])
    check_repr(SparseMatrix(indices=[[1, 2], [3, 4]], data=[5, 6], shape=(7, 8)))
    assert repr(SparseMatrix(((1, 2), (3, 4)), (5, 6), (7, 8))).replace(
        ", dtype=int64", ""
    ) == (
        "SparseMatrix(indices=array([[1, 2],\n       [3, 4]]), "
        "data=array([5, 6]), shape=(7, 8))"
    )

    check_init_args(ChannelShape, ["shape", "channels_last"])
    check_repr(ChannelShape(shape=(1, 2, 3), channels_last=True))
    assert (
        repr(ChannelShape((1, 2, 3)))
        == "ChannelShape(shape=(1, 2, 3), channels_last=True)"
    )
    assert (
        repr(ChannelShape((1, 2, 3), channels_last=False))
        == "ChannelShape(shape=(1, 2, 3), channels_last=False)"
    )

    # __str__ always has channels last
    assert str(ChannelShape((1, 2, 3))) == "(1, 2, ch=3)"
    assert str(ChannelShape((1, 2, 3), channels_last=False)) == "(ch=1, 2, 3)"

    check_init_args(NoTransform, ["size_in"])
    check_repr(NoTransform(size_in=1))
    for dimensions in range(2):
        assert repr(NoTransform(dimensions)) == f"NoTransform(size_in={dimensions})"


def test_signals():
    assert repr(Signal(np.array([0.0]))) == "Signal(name=None, shape=(1,))"
    assert (
        repr(Signal(np.array([1.0, 1.0]), name="one")) == "Signal(name=one, shape=(2,))"
    )


def test_operators():
    sig = Signal(np.array([0.0]), name="sig")
    assert fnmatch(repr(TimeUpdate(sig, sig)), "<TimeUpdate at 0x*>")
    assert fnmatch(repr(TimeUpdate(sig, sig, tag="tag")), "<TimeUpdate 'tag' at 0x*>")
    assert fnmatch(repr(Reset(sig)), "<Reset at 0x*>")
    assert fnmatch(repr(Reset(sig, tag="tag")), "<Reset 'tag' at 0x*>")
    assert fnmatch(repr(Copy(sig, sig)), "<Copy at 0x*>")
    assert fnmatch(repr(Copy(sig, sig, tag="tag")), "<Copy 'tag' at 0x*>")
    assert fnmatch(repr(ElementwiseInc(sig, sig, sig)), "<ElementwiseInc at 0x*>")
    assert fnmatch(
        repr(ElementwiseInc(sig, sig, sig, tag="tag")), "<ElementwiseInc 'tag' at 0x*>"
    )
    assert fnmatch(repr(DotInc(sig, sig, sig)), "<DotInc at 0x*>")
    assert fnmatch(repr(DotInc(sig, sig, sig, tag="tag")), "<DotInc 'tag' at 0x*>")
    assert fnmatch(repr(SimPyFunc(sig, lambda x: 0.0, True, sig)), "<SimPyFunc at 0x*>")
    assert fnmatch(
        repr(SimPyFunc(sig, lambda x: 0.0, True, sig, tag="tag")),
        "<SimPyFunc 'tag' at 0x*>",
    )
    assert fnmatch(repr(SimPES(sig, sig, sig, 0.1)), "<SimPES at 0x*>")
    assert fnmatch(repr(SimPES(sig, sig, sig, 0.1, tag="tag")), "<SimPES 'tag' at 0x*>")
    assert fnmatch(repr(SimBCM(sig, sig, sig, sig, 0.1)), "<SimBCM at 0x*>")
    assert fnmatch(
        repr(SimBCM(sig, sig, sig, sig, 0.1, tag="tag")), "<SimBCM 'tag' at 0x*>"
    )
    assert fnmatch(repr(SimOja(sig, sig, sig, sig, 0.1, 1.0)), "<SimOja at 0x*>")
    assert fnmatch(
        repr(SimOja(sig, sig, sig, sig, 0.1, 1.0, tag="tag")), "<SimOja 'tag' at 0x*>"
    )
    assert fnmatch(repr(SimVoja(sig, sig, sig, sig, 1.0, sig, 1.0)), "<SimVoja at 0x*>")
    assert fnmatch(
        repr(SimVoja(sig, sig, sig, sig, 0.1, sig, 1.0, tag="tag")),
        "<SimVoja 'tag' at 0x*>",
    )
    assert fnmatch(repr(SimRLS(sig, sig, sig, sig)), "<SimRLS at 0x*>")
    assert fnmatch(
        repr(SimRLS(sig, sig, sig, sig, tag="tag")),
        "<SimRLS 'tag' at 0x*>",
    )
    assert fnmatch(repr(SimNeurons(LIF(), sig, {"sig": sig})), "<SimNeurons at 0x*>")
    assert fnmatch(
        repr(SimNeurons(LIF(), sig, {"sig": sig}, tag="tag")),
        "<SimNeurons 'tag' at 0x*>",
    )
    assert fnmatch(repr(SimProcess(WhiteNoise(), sig, sig, sig)), "<SimProcess at 0x*>")
    assert fnmatch(
        repr(SimProcess(WhiteNoise(), sig, sig, sig, tag="tag")),
        "<SimProcess 'tag' at 0x*>",
    )
