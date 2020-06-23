import pytest

import nengo
import nengo.synapses
from nengo.config import SupportDefaultsMixin
from nengo.exceptions import ConfigError, ReadonlyError
from nengo.params import Default, Parameter, Unconfigurable
from nengo.utils.testing import ThreadedAssertion


def test_config_basic():
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param("something", Parameter("something", None))
    model.config[nengo.Ensemble].set_param("other", Parameter("other", default=0))
    model.config[nengo.Connection].set_param(
        "something_else", Parameter("something_else", None)
    )

    with pytest.raises(ConfigError, match="'fails' is not a parameter"):
        model.config[nengo.Ensemble].set_param("fails", 1.0)

    with pytest.raises(ConfigError, match="'bias' is already a parameter in"):
        model.config[nengo.Ensemble].set_param("bias", Parameter("bias", None))

    with model:
        a = nengo.Ensemble(50, dimensions=1)
        b = nengo.Ensemble(90, dimensions=1)
        a2b = nengo.Connection(a, b, synapse=0.01)

    with pytest.raises(ConfigError, match="Cannot set parameters on an instance"):
        model.config[a].set_param("thing", Parameter("thing", None))

    with pytest.raises(ConfigError, match="Cannot get parameters on an instance"):
        model.config[a].get_param("bias")

    assert model.config[a].something is None
    assert model.config[b].something is None
    assert model.config[a].other == 0
    assert model.config[b].other == 0
    assert model.config[a2b].something_else is None

    model.config[a].something = "hello"
    assert model.config[a].something == "hello"
    model.config[a].something = "world"
    assert model.config[a].something == "world"
    del model.config[a].something
    assert model.config[a].something is None

    with pytest.raises(AttributeError):
        model.config[a].something_else
    with pytest.raises(AttributeError):
        model.config[a2b].something
    with pytest.raises(AttributeError):
        model.config[a].something_else = 1
    with pytest.raises(AttributeError):
        model.config[a2b].something = 1

    with pytest.raises(ConfigError, match="'str' is not set up for configuration"):
        model.config["a"].something
    with pytest.raises(ConfigError, match="'NoneType' is not set up for configuration"):
        model.config[None].something


def test_network_nesting():
    """Make sure nested networks inherit configs."""
    with nengo.Network() as net1:
        # We'll change radius and seed. Make sure we have the right defaults.
        assert nengo.Ensemble.radius.default == 1.0
        assert nengo.Ensemble.seed.default is None

        # Before = default; after = what we set
        assert net1.config[nengo.Ensemble].radius == 1.0
        assert net1.config[nengo.Ensemble].seed is None
        net1.config[nengo.Ensemble].radius = 3.0
        net1.config[nengo.Ensemble].seed = 10
        assert net1.config[nengo.Ensemble].radius == 3.0
        assert net1.config[nengo.Ensemble].seed == 10

        # If we make an ensemble, it uses what we set, of the config default
        ens1 = nengo.Ensemble(10, 1, radius=2.0)
        assert ens1.seed == 10
        assert ens1.radius == 2.0

        # It's an error to configure an actual param with the config
        with pytest.raises(ConfigError):
            net1.config[ens1].radius = 3.0

        with nengo.Network() as net2:
            # We'll just change radius in net2.

            # Before = default; after = what we set
            assert net2.config[nengo.Ensemble].radius == 1.0
            net2.config[nengo.Ensemble].radius = 5.0
            assert net2.config[nengo.Ensemble].radius == 5.0

            # If we make an ensemble, it traverses the context stack
            ens2 = nengo.Ensemble(10, 1)
            assert ens2.radius == 5.0
            assert ens2.seed == 10

            with nengo.Network() as net3:
                # Works for > 1 levels
                net3.config[nengo.Ensemble].seed = 20
                ens3 = nengo.Ensemble(10, 1)
                assert ens3.seed == 20
                assert ens3.radius == 5.0


def test_context_is_threadsafe():
    class CheckIndependence(ThreadedAssertion):
        def init_thread(self, worker):
            setattr(worker, "model", nengo.Network())
            worker.model.__enter__()

        def assert_thread(self, worker):
            assert list(nengo.Config.context) == [worker.model.config]

        def finish_thread(self, worker):
            worker.model.__exit__(*worker.exc_info)

    CheckIndependence(n_threads=2).run()


def test_defaults():
    """Test that settings defaults propagates appropriately."""
    b = nengo.Ensemble(10, dimensions=1, radius=nengo.Default, add_to_container=False)

    assert b.radius == nengo.Ensemble.radius.default

    with nengo.Network():
        c = nengo.Ensemble(10, dimensions=1, radius=nengo.Default)
        with nengo.Network() as net2:
            net2.config[nengo.Ensemble].radius = 2.0
            a = nengo.Ensemble(50, dimensions=1, radius=nengo.Default)
            del net2.config[nengo.Ensemble].radius
            d = nengo.Ensemble(50, dimensions=1, radius=nengo.Default)

    assert c.radius == nengo.Ensemble.radius.default
    assert a.radius == 2.0
    assert d.radius == nengo.Ensemble.radius.default


def test_configstack():
    """Test that setting defaults with bare configs works."""
    with nengo.Network() as net:
        net.config[nengo.Connection].transform = -1
        e1 = nengo.Ensemble(5, dimensions=1)
        e2 = nengo.Ensemble(6, dimensions=1)
        excite = nengo.Connection(e1, e2)
        with nengo.Config(nengo.Connection) as inhib:
            inhib[nengo.Connection].synapse = nengo.synapses.Lowpass(0.00848)
            inhibit = nengo.Connection(e1, e2)
    assert excite.synapse == nengo.Connection.synapse.default
    assert excite.transform.init == -1
    assert inhibit.synapse == inhib[nengo.Connection].synapse
    assert inhibit.transform.init == -1


def test_config_property():
    """Test that config can't be easily modified."""
    with nengo.Network() as net:
        with pytest.raises(ReadonlyError):
            net.config = nengo.config.Config()
        with pytest.raises(AttributeError):
            del net.config
        assert nengo.config.Config.context[-1] is net.config
    assert len(nengo.config.Config.context) == 0


def test_external_class():
    class A:
        thing = Parameter("thing", default="hey")

    inst = A()
    config = nengo.Config(A)
    config[A].set_param("amount", Parameter("amount", default=1))

    # Extra param
    assert config[inst].amount == 1

    # Default still works like Nengo object
    assert inst.thing == "hey"
    with pytest.raises(ConfigError):
        config[inst].thing


def test_instance_fallthrough():
    """If the class default is set, instances should use that."""

    class A:
        pass

    inst1 = A()
    inst2 = A()
    config = nengo.Config(A)
    config[A].set_param("amount", Parameter("amount", default=1))
    assert config[A].amount == 1
    assert config[inst1].amount == 1
    assert config[inst2].amount == 1
    # Value can change for instance
    config[inst1].amount = 2
    assert config[A].amount == 1
    assert config[inst1].amount == 2
    assert config[inst2].amount == 1
    # If value to A is changed, unset instances should also change
    config[A].amount = 3
    assert config[A].amount == 3
    assert config[inst1].amount == 2
    assert config[inst2].amount == 3
    # If class default is deleted, unset instances go back
    del config[A].amount
    assert config[A].amount == 1
    assert config[inst1].amount == 2
    assert config[inst2].amount == 1


def test_contains():
    class A:
        pass

    cfg = nengo.Config(A)
    with pytest.raises(TypeError, match="Cannot check if .* is in a config"):
        A in cfg

    net = nengo.Network()

    net.config[nengo.Ensemble].set_param("test", Parameter("test", None))
    assert "test" not in net.config[nengo.Ensemble]

    net.config[nengo.Ensemble].test = "testval"
    assert "test" in net.config[nengo.Ensemble]


def test_subclass_config():
    class MyParent(SupportDefaultsMixin):
        p = Parameter("p", default="baba")

        def __init__(self, p=Default):
            self.p = p

    class MyChild(MyParent):
        pass

    with nengo.Config(MyParent) as cfg:
        cfg[MyParent].p = "value1"
        a = MyChild()
        assert a.p == "value1"

    with nengo.Config(MyParent) as cfg:
        cfg[MyChild].p = "value2"
        a = MyChild()
        assert a.p == "value2"

    # If any config entry in the current context fits with the object being
    # instantiated, we use that entry, even if there's an entry that's a
    # "better" fit (i.e. same class) in a higher context.
    with nengo.Config(MyParent) as cfg1:
        cfg1[MyChild].p = "value1"

        with nengo.Config(MyParent) as cfg2:
            cfg2[MyParent].p = "value2"
            a = MyChild()
            assert a.p == "value2"


def test_classparams_del():
    """tests ClassParams.__delattr__"""
    net = nengo.Network()
    clsparams = net.config[nengo.Ensemble]

    # test that normal instance param can be added/deleted
    clsparams.set_param("test", Parameter("test", None))
    clsparams.test = "val"
    assert "test" in clsparams
    del clsparams.test
    assert "test" not in clsparams

    # test that we can set/get/delete underscore attributes regularly
    clsparams._test = 3
    assert hasattr(clsparams, "_test") and clsparams._test == 3
    del clsparams._test
    assert not hasattr(clsparams, "_test")


def test_classparams_str_repr():
    """tests the repr function in classparams class"""
    clsparams = nengo.Network().config[nengo.Ensemble]
    clsparams.set_param("test", Parameter("test", None))
    assert repr(clsparams) == "<ClassParams[Ensemble]{test: None}>"
    assert str(clsparams) == "No parameters configured for Ensemble."

    clsparams.test = "val"
    assert str(clsparams) == "Parameters configured for Ensemble:\n  test: val"


def test_config_repr():
    """tests the repr function in Config class"""
    model = nengo.Network()
    r = repr(model.config)  # == "<Config(Connection, Ensemble, Node, Probe)>
    assert (  # Python <= 3.5 has types in any order
        r.startswith("<Config(")
        and r.endswith(")>")
        and "Connection" in r
        and "Ensemble" in r
        and "Node" in r
        and "Probe" in r
    )


def test_config_exit_errors():
    """Tests ConfigErrors in `Config.__exit__`"""
    model = nengo.Network()
    with pytest.raises(
        ConfigError, match="Config.context in bad state; was empty when exiting"
    ):
        model.config.__exit__(0, 0, 0)

    model2 = nengo.Network()
    with pytest.raises(
        ConfigError, match="Config.context in bad state; was expecting current"
    ):
        model2.config.__enter__()
        model.config.__exit__(0, 0, 0)


def test_instanceparams_str_repr():
    """Test the str and repr functions for InstanceParams class"""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(50, dimensions=1, label="a")
        model.config[nengo.Ensemble].set_param("prm", nengo.params.Parameter("prm"))
        model.config[a].prm = "val"

    assert str(model.config[a]) == 'Parameters set for <Ensemble "a">:\n  prm: val'
    assert repr(model.config[a]) == '<InstanceParams[<Ensemble "a">]{prm: val}>'


def test_instanceparams_contains():
    """tests the contains function in InstanceParams class"""
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param("test", Parameter("test", None))
    with model:
        a = nengo.Ensemble(5, 1)
        b = nengo.Ensemble(5, 1)
        model.config[b].test = 3

    assert "test" not in model.config[a]
    assert "test" in model.config[b]


def test_instanceparams_del():
    """tests built in params on the delattr function in InstanceParams class"""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(5, dimensions=1)

    # test that normal instance param can be added/deleted
    model.config[nengo.Ensemble].set_param("test", Parameter("test", None))
    model.config[a].test = "val"
    assert "test" in model.config[a]
    del model.config[a].test
    assert "test" not in model.config[a]

    # test that built-in parameter cannot be deleted
    with pytest.raises(ConfigError, match="Cannot configure the built-in parameter"):
        del model.config[a].bias

    # test that we can set/get/delete underscore attributes regularly
    model.config[a]._test = 3
    assert hasattr(model.config[a], "_test") and model.config[a]._test == 3
    del model.config[a]._test
    assert not hasattr(model.config[a], "_test")


def test_unconfigurable_configerror():
    """Tests exception when using settattr on something that is not configurable"""
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param("prm", Parameter("prm", Unconfigurable))

    with pytest.raises(ConfigError, match="'prm' is not configurable"):
        model.config[nengo.Ensemble].prm = "other"


def test_reuse_parameters_configerror(request):
    """test that exception is raised when
    reusing parameters"""

    def finalizer():
        del nengo.Ensemble.same

    request.addfinalizer(finalizer)

    model = nengo.Network()
    nengo.Ensemble.same = Parameter("param_a")
    with pytest.raises(ConfigError, match="'same' is already a parameter in"):
        model.config[nengo.Ensemble].set_param("same", Parameter("param_b"))


def test_no_configures_args_error():
    with pytest.raises(TypeError, match="configures.* takes 1 or more arguments"):
        nengo.Network().config.configures()


def test_unconfigurable_default_configerror(request):
    """test exception when using `Config.default` with a unconfigurable parameter"""

    def finalizer():
        del nengo.Ensemble.something2

    request.addfinalizer(finalizer)

    model = nengo.Network()
    nengo.Ensemble.something2 = Parameter("something2", Unconfigurable)

    with pytest.raises(ConfigError, match="Unconfigurable parameters have no defaults"):
        model.config.default(nengo.Ensemble, "something2")


def test_get_param_on_instance_configerror():
    """test that exception is raised when getting params on instance"""
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param("something", Parameter("something", None))
    with model:
        a = nengo.Ensemble(50, dimensions=1)

    with pytest.raises(ConfigError, match="Cannot get parameters on an instance"):
        model.config[a].get_param("something")


@pytest.mark.parametrize("simplified", [False, True])
def test_supportdefaultsmixin_exceptions(request, simplified):
    def finalizer(val=nengo.rc.get("exceptions", "simplified")):
        nengo.rc.set("exceptions", "simplified", val)

    request.addfinalizer(finalizer)

    nengo.rc.set("exceptions", "simplified", str(simplified))

    class MyClass(SupportDefaultsMixin):
        p = Parameter("p", default="baba", readonly=True)

        def __init__(self, p=Default):
            self.p = p

    obj = MyClass()
    with pytest.raises(ReadonlyError, match="p is read-only and cannot be changed"):
        obj.p = "newval"
