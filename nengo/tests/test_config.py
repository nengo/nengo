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

    with pytest.raises(ConfigError):
        model.config[nengo.Ensemble].set_param("fails", 1.0)

    with model:
        a = nengo.Ensemble(50, dimensions=1)
        b = nengo.Ensemble(90, dimensions=1)
        a2b = nengo.Connection(a, b, synapse=0.01)

    with pytest.raises(ConfigError):
        model.config[a].set_param("thing", Parameter("thing", None))

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
        model.config[a2b].something
    with pytest.raises(AttributeError):
        model.config[a].something_else = 1
        model.config[a2b].something = 1

    with pytest.raises(ConfigError):
        model.config["a"].something
    with pytest.raises(ConfigError):
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

    CheckIndependence(n_threads=2)


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
    """tests the contains function"""

    class A:
        pass

    cfg = nengo.Config(A)
    with pytest.raises(TypeError):
        A in cfg

    model = nengo.Network()

    model.config[nengo.Ensemble].set_param("test2", Parameter("test2", None))

    assert model.config[nengo.Ensemble].__contains__("test2") is False


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


def test_classparams_repr():
    """tests the repr function in classparams class"""
    model = nengo.Network()
    with model:
        model.config[nengo.Ensemble].set_param("containstest", Parameter("test", None))
        assert (
            repr(model.config[nengo.Ensemble])
            == "<ClassParams[Ensemble]{containstest: None}>"
        )


def test_config_repr():
    """tests the repr function in Config class"""
    model = nengo.Network()
    with model:  # == "<Config(Connection, Ensemble, Node, Probe)>
        assert (
            repr(model.config).startswith("<Config(")
            and repr(model.config).endswith(")>")
            and "Connection" in repr(model.config)
            and "Ensemble" in repr(model.config)
            and "Node" in repr(model.config)
            and "Probe" in repr(model.config)
        )
        # travis-ci has it in any order


def test_config_exit():
    """tests the exit function in Config class"""
    model = nengo.Network()
    model2 = nengo.Network()
    with model:
        with pytest.raises(ConfigError):
            model.config.__enter__()
            model2.config.__exit__(0, 0, 0)

    with pytest.raises(ConfigError):
        with (model.config):
            model.config.context.clear()


def test_instanceparams_str():
    """tests the str function in InstanceParams class"""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(50, dimensions=1, label="test")
    assert str(model.config[a]) == 'Parameters set for <Ensemble "test">:'
    # The right type of params are created on init, how do I edit those?
    # TODO: figure out why this isn't working


def test_instanceparams_repr():
    """tests the repr function in InstanceParams class"""
    model = nengo.Network()
    with model:
        a = nengo.Ensemble(50, dimensions=1, label="test")
        # model.config[nengo.Ensemble].set_param("test", None)
        model.config[nengo.Ensemble].set_param("test", nengo.params.Parameter("test"))
        model.config[a].test = "test"
        # Cannot set parameters on an instance, how do I fill up self._clsparams.params
    assert repr(model.config[a]) == '<InstanceParams[<Ensemble "test">]{test: test}>'


def test_instanceparams_contains():
    """tests the contains function in InstanceParams class"""
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param("containstest", Parameter("test", None))
    with model:
        a = nengo.Ensemble(50, dimensions=1)
    "containstest" in model.config[a]


def test_instanceparams_delattr_configerror():
    """tests built in params on the delattr function in InstanceParams class"""
    with pytest.raises(ConfigError):
        model = nengo.Network()
        with model:
            a = nengo.Ensemble(50, dimensions=1)
        model.config[a].__delattr__("bias")


def test_underscore_del_instance_params_exception():
    """tests that exception is raised for instance params that start with underscore"""
    with pytest.raises(Exception):
        model = nengo.Network()
        model.config[nengo.Ensemble].set_param("_uscore", Parameter("_uscore", None))
        with model:
            a = nengo.Ensemble(50, dimensions=1)
        del model.config[a]._uscore


def test_not_configurable_configerror():
    """tests that exception is raised when
    using settattr on something that is not configurable"""
    with pytest.raises(ConfigError):
        model = nengo.Network()
        my_param = Parameter("something", Unconfigurable)

        model.config[nengo.Ensemble].set_param("something", my_param)
        model.config[nengo.Ensemble].something = "other"


def test_underscore_del_class_params_exception(request):
    """test that exception is raised when
    using parameters that start with underscore in class params"""
    model = nengo.Network()
    my_param = Parameter("_uscore")

    model.config[nengo.Ensemble].set_param("_uscore", my_param)

    model.config[nengo.Ensemble]._uscore = "other"

    del model.config[nengo.Ensemble]._uscore


def test_reuse_parameters_configerror(request):
    """test that exception is raised when
    reusing parameters"""

    def finalizer():
        del nengo.Ensemble.same

    request.addfinalizer(finalizer)

    with pytest.raises(ConfigError):
        model = nengo.Network()
        nengo.Ensemble.same = Parameter("param3")
        model.config[nengo.Ensemble].set_param("same", Parameter("param2"))


def test_no_configures_error():
    with pytest.raises(TypeError):
        model = nengo.Network()
        classes = []
        assert len(classes) == 0
        model.config.configures(*classes)


def test_not_configurable_config_configerror(request):
    """test that exception is raised when
    using default on config class with non configurable"""

    def finalizer():
        del nengo.Ensemble.something2

    request.addfinalizer(finalizer)
    with pytest.raises(ConfigError):
        model = nengo.Network()
        my_param = Parameter("something2", Unconfigurable)

        nengo.Ensemble.something2 = my_param
        model.config.default(nengo.Ensemble, "something2")


def test_get_param_on_instance_configerror():
    """test that exception is raised when getting params on instance"""
    with pytest.raises(Exception):
        model = nengo.Network()
        model.config[nengo.Ensemble].set_param(
            "something", Parameter("something", None)
        )
        with model:
            a = nengo.Ensemble(50, dimensions=1)
        model.config[a].get_param("something")


def test_repr_on_param():
    """test repr for a param"""
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param("test", Parameter("test", None))
    with model:
        a = nengo.Ensemble(50, dimensions=1)
    model.config[a].test = "Hello"
    assert repr(model.config[a].test) == "'Hello'"


def test_reuse_default_param_configerror():
    """test that exception is raised when reusing default params"""
    with pytest.raises(Exception):
        model = nengo.Network()
        model.config[nengo.Ensemble].set_param("bias", Parameter("bias", None))
