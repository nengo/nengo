from io import BytesIO

import pytest

import nengo
from nengo.exceptions import (
    BuildError,
    CacheIOError,
    ConfigError,
    FingerprintError,
    MovedError,
    NetworkContextError,
    ObsoleteError,
    ReadonlyError,
    SignalError,
    SimulationError,
    SimulatorClosed,
    SpaModuleError,
    SpaParseError,
    Unconvertible,
    ValidationError,
)
from nengo.params import ObsoleteParam
from nengo.rc import rc, RC_DEFAULTS
import nengo.spa
from nengo.utils.builder import generate_graphviz


def check_tb_entries(entries, name_statement_pairs):
    assert len(entries) == len(name_statement_pairs)
    for entry, (name, statement) in zip(entries, name_statement_pairs):
        assert entry.name == name
        if statement.endswith("..."):
            assert str(entry.statement.deindent()).startswith(statement[:-3])
        else:
            assert str(entry.statement.deindent()) == statement


def test_validation_error(request):
    # Ensure settings are set back to default after the test, even if it fails
    request.addfinalizer(
        lambda: rc.set(
            "exceptions", "simplified", str(RC_DEFAULTS["exceptions"]["simplified"])
        )
    )

    nengo.rc["exceptions"]["simplified"] = "False"

    with nengo.Network():
        with pytest.raises(ValidationError) as excinfo:
            nengo.Ensemble(n_neurons=0, dimensions=1)

    assert str(excinfo.value) == (
        "Ensemble.n_neurons: Value must be greater than or equal to 1 (got 0)"
    )
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_validation_error", "nengo.Ensemble(n_neurons=0, dimensions=1)"),
            ("__call__", "inst.__init__(*args, **kwargs)"),
            ("__init__", "self.n_neurons = n_neurons"),
            ("__setattr__", "super().__setattr__(name, val)"),
            ("__setattr__", "super().__setattr__(name, val)"),
            ("__set__", "self.data[instance] = self.coerce(instance, value)"),
            ("coerce", "return super().coerce(instance, num)"),
            ("coerce", "raise ValidationError(..."),
        ],
    )

    nengo.rc["exceptions"]["simplified"] = "True"

    with pytest.raises(ValidationError) as excinfo:
        nengo.dists.PDF(x=[1, 1], p=[0.1, 0.2])

    assert str(excinfo.value) == "PDF.p: PDF must sum to one (sums to 0.300000)"
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_validation_error", "nengo.dists.PDF(x=[1, 1], p=[0.1, 0.2])"),
            ("__init__", "raise ValidationError(..."),
        ],
    )


def test_readonly_error():
    with nengo.Network():
        ens = nengo.Ensemble(n_neurons=10, dimensions=1)
        p = nengo.Probe(ens)
        with pytest.raises(ReadonlyError) as excinfo:
            p.target = ens
    assert (
        str(excinfo.value) == "Probe.target: target is read-only and cannot be changed"
    )

    class Frozen(nengo.params.FrozenObject):
        p = nengo.params.Parameter("p", readonly=False)

    with pytest.raises(ReadonlyError) as excinfo:
        Frozen()

    assert str(excinfo.value).endswith(
        "All parameters of a FrozenObject must be readonly"
    )
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_readonly_error", "Frozen()"),
            ("__init__", "raise ReadonlyError(attr=p, obj=self, msg=msg)",),
        ],
    )


def test_build_error():
    model = nengo.builder.Model()
    with pytest.raises(BuildError) as excinfo:
        nengo.builder.Builder.build(model, "")

    assert str(excinfo.value) == "Cannot build object of type 'str'"
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_build_error", 'nengo.builder.Builder.build(model, "")'),
            (
                "build",
                'raise BuildError("Cannot build object of type %r" % '
                "type(obj).__name__)",
            ),
        ],
    )


def test_obsolete_error():
    class Test:
        ab = ObsoleteParam("ab", "msg")

    inst = Test()
    with pytest.raises(ObsoleteError) as excinfo:
        print(inst.ab)

    assert str(excinfo.value) == "Obsolete: msg"
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_obsolete_error", "print(inst.ab)"),
            ("__get__", "self.raise_error()"),
            (
                "raise_error",
                "raise ObsoleteError(self.short_msg, since=self.since, url=self.url)",
            ),
        ],
    )


def test_moved_error():
    with pytest.raises(MovedError) as excinfo:
        generate_graphviz()

    assert str(excinfo.value) == "This feature has been moved to nengo_extras.graphviz"
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_moved_error", "generate_graphviz()"),
            ("generate_graphviz", 'raise MovedError(location="nengo_extras.graphviz")'),
        ],
    )


def test_config_error():
    with pytest.raises(ConfigError) as excinfo:
        print(nengo.Network().config[object])

    assert str(excinfo.value) == (
        "Type 'object' is not set up for configuration. "
        "Call 'configures(object)' first."
    )
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_config_error", "print(nengo.Network().config[object])"),
            ("__getitem__", "raise ConfigError(..."),
        ],
    )


def test_spa_module_error():
    with pytest.raises(SpaModuleError) as excinfo:
        with nengo.spa.SPA():
            nengo.spa.State(1, label="1")

    assert (
        str(excinfo.value) == '<State "1"> must be set as an attribute of a SPA network'
    )
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_spa_module_error", 'nengo.spa.State(1, label="1")'),
            ("__exit__", "raise SpaModuleError(..."),
        ],
    )


def test_spa_parse_error():
    vocab = nengo.spa.Vocabulary(16)
    with pytest.raises(SpaParseError) as excinfo:
        print(vocab["a"])

    assert str(excinfo.value) == "Semantic pointers must begin with a capital letter."
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_spa_parse_error", 'print(vocab["a"])'),
            (
                "__getitem__",
                'raise SpaParseError("Semantic pointers must begin with a capital '
                'letter.")',
            ),
        ],
    )


def test_simulator_closed():
    with nengo.Network() as net:
        nengo.Ensemble(10, 1)
    with nengo.Simulator(net) as sim:
        sim.run(0.01)
    with pytest.raises(SimulatorClosed) as excinfo:
        sim.run(0.01)

    assert str(excinfo.value) == "Simulator cannot run because it is closed."
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_simulator_closed", "sim.run(0.01)"),
            ("run", "self.run_steps(steps, progress_bar=progress_bar)"),
            ("run_steps", "self.step()"),
            (
                "step",
                'raise SimulatorClosed("Simulator cannot run because it is closed.")',
            ),
        ],
    )


def test_simulation_error():
    with nengo.Network() as net:
        nengo.Node(lambda t: None if t > 0.002 else 1.0)
    with nengo.Simulator(net) as sim:
        with pytest.raises(SimulationError) as excinfo:
            sim.run(0.003)

    assert str(excinfo.value) == "Function '<lambda>' returned non-finite value"
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_simulation_error", "sim.run(0.003)"),
            ("run", "self.run_steps(steps, progress_bar=progress_bar)"),
            ("run_steps", "self.step()"),
            ("step", "step_fn()"),
            ("step_simpyfunc", "raise SimulationError(..."),
        ],
    )


def test_signal_error():
    s = nengo.builder.signal.Signal([1])
    with pytest.raises(SignalError) as excinfo:
        s.initial_value = 0

    assert str(excinfo.value) == "Cannot change initial value after initialization"
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_signal_error", "s.initial_value = 0"),
            (
                "initial_value",
                'raise SignalError("Cannot change initial value after initialization")',
            ),
        ],
    )


def test_fingerprint_error():
    with pytest.raises(FingerprintError) as excinfo:
        nengo.cache.Fingerprint(lambda x: x)

    assert str(excinfo.value) == "Object of type 'function' cannot be fingerprinted."
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_fingerprint_error", "nengo.cache.Fingerprint(lambda x: x)"),
            ("__init__", "raise FingerprintError(..."),
        ],
    )


def test_network_context_error(request):
    request.addfinalizer(nengo.Network.context.clear)

    with pytest.raises(NetworkContextError) as excinfo:
        with nengo.Network(label="net"):
            nengo.Network.context.append("bad")
            nengo.Ensemble(10, 1)

    assert str(excinfo.value) == (
        "Network.context in bad state; was expecting current context to be "
        "'<Network \"net\">' but instead got 'bad'."
    )
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_network_context_error", "nengo.Ensemble(10, 1)"),
            ("__exit__", "raise NetworkContextError(..."),
        ],
    )


def test_unconvertible():
    with nengo.Network() as net:
        n = nengo.Node(output=None, size_in=1)
        nengo.Connection(n, n, synapse=None)

    with pytest.raises(Unconvertible) as excinfo:
        nengo.utils.builder.remove_passthrough_nodes(net.nodes, net.connections)

    assert str(excinfo.value) == "Cannot remove a Node with a feedback connection"
    check_tb_entries(
        excinfo.traceback,
        [
            (
                "test_unconvertible",
                "nengo.utils.builder.remove_passthrough_nodes(net.nodes, "
                "net.connections)",
            ),
            ("remove_passthrough_nodes", "raise Unconvertible(..."),
        ],
    )


def test_cache_io_error():
    bio = BytesIO(b"a" * 40)
    with pytest.raises(CacheIOError) as excinfo:
        nengo.utils.nco.read(bio)

    assert str(excinfo.value) == "Not a Nengo cache object file."
    check_tb_entries(
        excinfo.traceback,
        [
            ("test_cache_io_error", "nengo.utils.nco.read(bio)"),
            ("read", 'raise CacheIOError("Not a Nengo cache object file.")'),
        ],
    )
