from fnmatch import fnmatch

import numpy as np

import nengo
from nengo import (
    AdaptiveLIF,
    AdaptiveLIFRate,
    Alpha,
    BCM,
    Direct,
    Izhikevich,
    LIF,
    LIFRate,
    LinearFilter,
    Lowpass,
    Oja,
    PES,
    RectifiedLinear,
    Sigmoid,
    Voja,
)
from nengo.builder.learning_rules import SimBCM, SimOja, SimPES
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
    Choice,
    Gaussian,
    PDF,
    SqrtBeta,
    SubvectorLength,
    Uniform,
    UniformHypersphere,
)
from nengo.processes import BrownNoise, FilteredNoise, WhiteNoise, WhiteSignal
from nengo.synapses import Triangle


def test_core_objects():
    with nengo.Network() as net1:

        a = nengo.Ensemble(100, 2)
        with nengo.Network(label="net2") as net2:
            b = nengo.Ensemble(100, 2, label="b")

        ap = nengo.Probe(a)
        bp = nengo.Probe(b)

        c1 = nengo.Connection(a, b)
        c2 = nengo.Connection(a, b, function=np.square)

        n1 = nengo.Node(output=np.sin)
        n2 = nengo.Node(output=np.cos, label="n2")

    assert fnmatch(repr(net1), "<Network (unlabeled) at 0x*>")
    assert fnmatch(repr(net2), '<Network "net2" at 0x*>')

    # Ensemble
    assert fnmatch(repr(a), "<Ensemble (unlabeled) at 0x*>")
    assert fnmatch(repr(b), '<Ensemble "b" at 0x*>')

    # Probe
    assert fnmatch(
        repr(ap), "<Probe at 0x* of 'decoded_output' of <Ensemble (unlabeled) at 0x*>>"
    )
    assert fnmatch(repr(bp), "<Probe at 0x* of 'decoded_output' of <Ensemble \"b\">>")

    # Connection
    assert fnmatch(
        repr(c1),
        '<Connection at 0x* from <Ensemble (unlabeled) at 0x*> to <Ensemble "b">>',
    )
    assert fnmatch(
        repr(c2),
        '<Connection at 0x* from <Ensemble (unlabeled) at 0x*> to <Ensemble "b"> '
        "computing 'square'>",
    )

    # Node
    assert fnmatch(repr(n1), "<Node (unlabeled) at 0x*>")
    assert fnmatch(repr(n2), '<Node "n2" at 0x*>')

    # Neurons
    assert fnmatch(repr(b.neurons), '<Neurons at 0x* of <Ensemble "b" at 0x*>>')

    # ObjView
    assert fnmatch(repr(a[:1]), "<Ensemble (unlabeled) at 0x*>[[]:1[]]")


def test_neuron_types():
    assert repr(Direct()) == "Direct()"
    assert repr(RectifiedLinear()) == "RectifiedLinear()"
    assert repr(Sigmoid()), "Sigmoid()"
    assert repr(Sigmoid(tau_ref=0.001)) == "Sigmoid(tau_ref=0.001)"
    assert repr(LIFRate()) == "LIFRate()"
    assert repr(LIFRate(tau_rc=0.01, tau_ref=0)) == "LIFRate(tau_rc=0.01, tau_ref=0)"
    assert repr(LIF()) == "LIF()"
    assert repr(LIF(tau_rc=0.01, tau_ref=0)) == "LIF(tau_rc=0.01, tau_ref=0)"
    assert repr(AdaptiveLIFRate()) == "AdaptiveLIFRate()"
    assert (
        repr(AdaptiveLIFRate(tau_rc=0.01, tau_n=0.5, inc_n=0.02))
        == "AdaptiveLIFRate(tau_n=0.5, inc_n=0.02, tau_rc=0.01)"
    )
    assert repr(AdaptiveLIF()) == "AdaptiveLIF()"
    assert (
        repr(AdaptiveLIF(tau_rc=0.01, tau_n=0.5, inc_n=0.02))
        == "AdaptiveLIF(tau_n=0.5, inc_n=0.02, tau_rc=0.01)"
    )
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


def test_learning_rule_types():
    assert repr(PES()) == "PES()"
    assert (
        repr(PES(learning_rate=1e-6, pre_synapse=0.01))
        == "PES(learning_rate=1e-06, pre_synapse=Lowpass(tau=0.01))"
    )
    assert repr(BCM()) == "BCM()"
    assert repr(
        BCM(
            learning_rate=1e-8,
            pre_synapse=0.01,
            post_synapse=0.005,
            theta_synapse=10.0,
        )
    ) == (
        "BCM(learning_rate=1e-08, pre_synapse=Lowpass(tau=0.01), "
        "post_synapse=Lowpass(tau=0.005), theta_synapse=Lowpass(tau=10.0))"
    )
    assert repr(Oja()) == "Oja()"
    assert repr(
        Oja(learning_rate=1e-5, pre_synapse=0.01, post_synapse=0.005, beta=0.5)
    ) == (
        "Oja(learning_rate=1e-05, pre_synapse=Lowpass(tau=0.01), "
        "post_synapse=Lowpass(tau=0.005), beta=0.5)"
    )

    assert repr(Voja()) == "Voja()"
    assert (
        repr(Voja(learning_rate=1e-5, post_synapse=None))
        == "Voja(learning_rate=1e-05, post_synapse=None)"
    )


def test_distributions():
    assert (
        repr(PDF([1, 2], [0.4, 0.6])) == "PDF(x=array([1., 2.]), p=array([0.4, 0.6]))"
    )
    assert repr(Uniform(0, 1)) == "Uniform(low=0, high=1)"
    assert repr(Uniform(0, 5, integer=True)) == "Uniform(low=0, high=5, integer=True)"
    assert repr(Gaussian(1, 0.1)) == "Gaussian(mean=1, std=0.1)"
    assert repr(UniformHypersphere()) == "UniformHypersphere()"
    assert repr(UniformHypersphere(surface=True)) == "UniformHypersphere(surface=True)"
    assert repr(Choice([1, 2, 3])) == "Choice(options=array([1., 2., 3.]))"
    assert (
        repr(Choice([1, 2, 3], weights=[0.1, 0.5, 0.4]))
        == "Choice(options=array([1., 2., 3.]), weights=array([0.1, 0.5, 0.4]))"
    )
    assert repr(SqrtBeta(3)) == "SqrtBeta(n=3)"
    assert repr(SqrtBeta(3, 2)) == "SqrtBeta(n=3, m=2)"
    assert repr(SubvectorLength(3)) == "SubvectorLength(dimensions=3)"


def test_synapses():
    assert repr(Lowpass(0.01)) == "Lowpass(tau=0.01)"
    assert repr(Alpha(0.02)) == "Alpha(tau=0.02)"
    assert repr(Triangle(0.03)) == "Triangle(t=0.03)"
    assert (
        repr(LinearFilter([1], [0.03, 1]))
        == "LinearFilter(num=array([1.]), den=array([0.03, 1.  ]))"
    )


def test_processes():
    gaussian = Gaussian(1, 2)
    assert (
        repr(WhiteNoise(gaussian, scale=False))
        == "WhiteNoise(dist=Gaussian(mean=1, std=2), scale=False)"
    )
    assert repr(FilteredNoise(Alpha(0.2), gaussian)) == (
        "FilteredNoise(synapse=Alpha(tau=0.2), dist=Gaussian(mean=1, std=2))"
    )
    assert repr(BrownNoise(gaussian)) == "BrownNoise(dist=Gaussian(mean=1, std=2))"
    assert (
        repr(WhiteSignal(0.2, 10, rms=0.3))
        == "WhiteSignal(period=0.2, high=10, rms=0.3)"
    )


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
    assert fnmatch(repr(SimPES(sig, sig, sig, sig, 0.1)), "<SimPES at 0x*>")
    assert fnmatch(
        repr(SimPES(sig, sig, sig, sig, 0.1, tag="tag")), "<SimPES 'tag' at 0x*>"
    )
    assert fnmatch(repr(SimBCM(sig, sig, sig, sig, 0.1)), "<SimBCM at 0x*>")
    assert fnmatch(
        repr(SimBCM(sig, sig, sig, sig, 0.1, tag="tag")), "<SimBCM 'tag' at 0x*>"
    )
    assert fnmatch(repr(SimOja(sig, sig, sig, sig, 0.1, 1.0)), "<SimOja at 0x*>")
    assert fnmatch(
        repr(SimOja(sig, sig, sig, sig, 0.1, 1.0, tag="tag")), "<SimOja 'tag' at 0x*>",
    )
    assert fnmatch(repr(SimNeurons(LIF(), sig, sig, [sig])), "<SimNeurons at 0x*>")
    assert fnmatch(
        repr(SimNeurons(LIF(), sig, sig, [sig], tag="tag")),
        "<SimNeurons 'tag' at 0x*>",
    )
    assert fnmatch(
        repr(SimProcess(WhiteNoise(), sig, sig, sig)), "<SimProcess at 0x*>",
    )
    assert fnmatch(
        repr(SimProcess(WhiteNoise(), sig, sig, sig, tag="tag")),
        "<SimProcess 'tag' at 0x*>",
    )
