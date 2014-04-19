import pytest

import nengo
from nengo.config import Parameter


def test_config_basic():
    config = nengo.Config()
    config.configure(nengo.Ensemble)
    config[nengo.Ensemble].set_param('something', Parameter(default=None))
    config[nengo.Ensemble].set_param('other', Parameter(default=0))
    config.configure(nengo.Connection)
    config[nengo.Connection].set_param('something_else', Parameter(None))

    model = nengo.Network()
    model.config = config
    with model:
        a = nengo.Ensemble(nengo.LIF(50), 1)
        b = nengo.Ensemble(nengo.LIF(90), 1)
        a2b = nengo.Connection(a, b, synapse=0.01)

    assert config[a].something is None
    assert config[b].something is None
    assert config[a].other == 0
    assert config[b].other == 0
    assert config[a2b].something_else is None

    config[a].something = 'hello'
    assert config[a].something == 'hello'
    config[a].something = 'world'
    assert config[a].something == 'world'

    with pytest.raises(AttributeError):
        config[a].something_else
        config[a2b].something
    with pytest.raises(AttributeError):
        config[a].something_else = 1
        config[a2b].something = 1

    with pytest.raises(KeyError):
        config['a'].something
    with pytest.raises(KeyError):
        config[None].something
    with pytest.raises(KeyError):
        config[model].something


def test_parameter_checking():
    class PositiveParameter(Parameter):
        def __set__(self, instance, value):
            if not isinstance(value, (int, float)) or value <= 0:
                raise AttributeError('value must be positive')
            super(PositiveParameter, self).__set__(instance, value)

    config = nengo.Config()
    config.configure(nengo.Ensemble)
    config[nengo.Ensemble].set_param('number', PositiveParameter(default=1))

    model = nengo.Network()
    model.config = config
    with model:
        a = nengo.Ensemble(50, 1)
        b = nengo.Ensemble(90, 1)

    config[a].number = 3
    with pytest.raises(AttributeError):
        config[a].number = 0
    with pytest.raises(AttributeError):
        config[b].number = 'a'


def test_config_parent():
    """Make sure passing a parent to Config works properly."""
    config = nengo.Config()
    config.configure(nengo.Ensemble)
    assert config[nengo.Ensemble].radius == 1.0

    # Change some params
    config[nengo.Ensemble].radius = 3.0
    config[nengo.Ensemble].seed = 10
    assert config[nengo.Ensemble].radius == 3.0
    assert nengo.Ensemble.radius.default == 1.0
    assert config[nengo.Ensemble].seed == 10
    assert nengo.Ensemble.seed.default is None

    # Propagates to new configs
    new_config = nengo.Config(parent=config)
    assert new_config[nengo.Ensemble].radius == 3.0
    assert new_config[nengo.Ensemble].seed == 10

    # But changing the new doesn't change the old
    new_config[nengo.Ensemble].radius = 5.0
    assert new_config[nengo.Ensemble].radius == 5.0
    assert config[nengo.Ensemble].radius == 3.0
    assert nengo.Ensemble.radius.default == 1.0

    # Many levels is fine
    new_new_config = nengo.Config(parent=new_config)
    assert new_new_config[nengo.Ensemble].radius == 5.0
    assert new_new_config[nengo.Ensemble].seed == 10


def test_network_nesting():
    """Make sure nested networks inherit configs."""
    with nengo.Network() as net1:
        assert net1.config[nengo.Ensemble].radius == 1.0
        net1.config[nengo.Ensemble].radius = 3.0
        net1.config[nengo.Ensemble].seed = 10
        assert net1.config[nengo.Ensemble].radius == 3.0
        assert nengo.Ensemble.radius.default == 1.0
        assert net1.config[nengo.Ensemble].seed == 10
        assert nengo.Ensemble.seed.default is None

        ens = nengo.Ensemble(nengo.LIF(10), 1, radius=2.0)
        assert net1.config[ens].radius == 3.0

        with nengo.Network() as net2:
            assert net2.config[nengo.Ensemble].radius == 3.0
            assert net2.config[nengo.Ensemble].seed == 10
            net2.config[nengo.Ensemble].radius = 5.0
            assert net2.config[nengo.Ensemble].radius == 5.0
            assert net1.config[nengo.Ensemble].radius == 3.0
            assert nengo.Ensemble.radius.default == 1.0

            with nengo.Network() as net3:
                assert net3.config[nengo.Ensemble].radius == 5.0
                assert net3.config[nengo.Ensemble].seed == 10


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
