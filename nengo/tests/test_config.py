import pytest

import nengo
import nengo.synapses
from nengo.exceptions import ConfigError, ReadonlyError
from nengo.params import Parameter
from nengo.utils.testing import ThreadedAssertion


def test_config_basic():
    model = nengo.Network()
    model.config[nengo.Ensemble].set_param('something',
                                           Parameter('something', None))
    model.config[nengo.Ensemble].set_param('other',
                                           Parameter('other', default=0))
    model.config[nengo.Connection].set_param('something_else',
                                             Parameter('something_else', None))

    with pytest.raises(ConfigError):
        model.config[nengo.Ensemble].set_param('fails', 1.0)

    with model:
        a = nengo.Ensemble(50, dimensions=1)
        b = nengo.Ensemble(90, dimensions=1)
        a2b = nengo.Connection(a, b, synapse=0.01)

    with pytest.raises(ConfigError):
        model.config[a].set_param('thing', Parameter('thing', None))

    assert model.config[a].something is None
    assert model.config[b].something is None
    assert model.config[a].other == 0
    assert model.config[b].other == 0
    assert model.config[a2b].something_else is None

    model.config[a].something = 'hello'
    assert model.config[a].something == 'hello'
    model.config[a].something = 'world'
    assert model.config[a].something == 'world'
    del model.config[a].something
    assert model.config[a].something is None

    with pytest.raises(AttributeError):
        model.config[a].something_else
        model.config[a2b].something
    with pytest.raises(AttributeError):
        model.config[a].something_else = 1
        model.config[a2b].something = 1

    with pytest.raises(ConfigError):
        model.config['a'].something
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
            setattr(worker, 'model', nengo.Network())
            worker.model.__enter__()

        def assert_thread(self, worker):
            assert list(nengo.Config.context) == [worker.model.config]

        def finish_thread(self, worker):
            worker.model.__exit__(*worker.exc_info)

    CheckIndependence(n_threads=2)


def test_defaults():
    """Test that settings defaults propagates appropriately."""
    b = nengo.Ensemble(10, dimensions=1, radius=nengo.Default,
                       add_to_container=False)

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
    assert excite.transform == -1
    assert inhibit.synapse == inhib[nengo.Connection].synapse
    assert inhibit.transform == -1


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
    class A(object):
        thing = Parameter('thing', default='hey')

    inst = A()
    config = nengo.Config(A)
    config[A].set_param('amount', Parameter('amount', default=1))

    # Extra param
    assert config[inst].amount == 1

    # Default still works like Nengo object
    assert inst.thing == 'hey'
    with pytest.raises(ConfigError):
        config[inst].thing


def test_instance_fallthrough():
    """If the class default is set, instances should use that."""
    class A(object):
        pass

    inst1 = A()
    inst2 = A()
    config = nengo.Config(A)
    config[A].set_param('amount', Parameter('amount', default=1))
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
