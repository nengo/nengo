import pytest

import nengo.config


def test_config():
    @nengo.config.configures(nengo.Ensemble)
    class TestConfigEnsemble(nengo.config.ConfigItem):
        something = nengo.config.Parameter(None)
        other = nengo.config.Parameter(0)

    @nengo.config.configures(nengo.Connection)
    class TestConfigConnection(nengo.config.ConfigItem):
        something_else = nengo.config.Parameter(None)

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(nengo.LIF(50), 1)
        b = nengo.Ensemble(nengo.LIF(90), 1)
        a2b = nengo.Connection(a, b, synapse=0.01)

    config = nengo.config.Config([TestConfigEnsemble, TestConfigConnection])

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
    class PositiveParameter(nengo.config.Parameter):
        def __set__(self, instance, value):
            if not isinstance(value, (int, float)) or value <= 0:
                raise AttributeError('value must be positive')
            super(PositiveParameter, self).__set__(instance, value)

    @nengo.config.configures(nengo.Ensemble)
    class TestConfigEnsemble(nengo.config.ConfigItem):
        number = PositiveParameter(1)

    model = nengo.Network()
    with model:
        a = nengo.Ensemble(50, 1)
        b = nengo.Ensemble(90, 1)

    config = nengo.config.Config([TestConfigEnsemble])

    config[a].number = 3
    with pytest.raises(AttributeError):
        config[a].number = 0
    with pytest.raises(AttributeError):
        config[b].number = 'a'


def test_invalid_config():

    @nengo.config.configures(nengo.Ensemble)
    class TestConfigEnsemble(nengo.config.ConfigItem):
        number = nengo.config.Parameter(1)

    class TestBadConfigConnection(nengo.config.ConfigItem):
        number = nengo.config.Parameter(1)

    with pytest.raises(TypeError):
        nengo.config.Config(None)
    with pytest.raises(AttributeError):
        nengo.config.Config([1, 2, 3])
    with pytest.raises(AttributeError):
        nengo.config.Config([TestBadConfigConnection])
    with pytest.raises(AttributeError):
        nengo.config.Config([TestConfigEnsemble, TestBadConfigConnection])


def test_defaults():
    """Test that settings defaults propagates appropriately."""
    b = nengo.Ensemble(nengo.LIF(10), 1, radius=nengo.Default,
                       add_to_container=False)

    assert b.radius == nengo.Config.context[0][nengo.Ensemble].radius

    with nengo.Network():
        c = nengo.Ensemble(nengo.LIF(10), 1, radius=nengo.Default)
        with nengo.Network() as net2:
            net2.config[nengo.Ensemble].radius = 2.0
            a = nengo.Ensemble(nengo.LIF(50), 1, radius=nengo.Default)

    assert c.radius == nengo.Config.context[0][nengo.Ensemble].radius
    assert a.radius == 2.0


def test_configstack():
    """Test that setting defaults with bare configs works."""
    @nengo.config.configures(nengo.Connection)
    class InhibitoryConnection(nengo.config.ConfigItem):
        synapse = nengo.config.Parameter(0.00848)

    inhib = nengo.Config([InhibitoryConnection])
    with nengo.Network():
        e1 = nengo.Ensemble(nengo.LIF(5), 1)
        e2 = nengo.Ensemble(nengo.LIF(6), 1)
        excite = nengo.Connection(e1, e2)
        with inhib:
            inhibit = nengo.Connection(e1, e2)
    assert excite.synapse == nengo.Config.context[0][nengo.Connection].synapse
    assert inhibit.synapse == inhib[nengo.Connection].synapse


def test_copy_depth():
    """Test that copy is deep enough"""
    with nengo.Network() as net1:
        net1.config[nengo.Ensemble].encoders = [[0]]
        with nengo.Network() as net2:
            net2.config[nengo.Ensemble].encoders = [[1]]

    assert net1.config[nengo.Ensemble].encoders == [[0]]


def test_copy_shallowness():
    """Test that copy is not too deep"""
    with nengo.Network() as net1:
        encoders = [[0]]
        net1.config[nengo.Ensemble].encoders = encoders
        with nengo.Network() as net2:
            encoders[0][0] = 1

    assert net1.config[nengo.Ensemble].encoders == [[1]]
    assert net2.config[nengo.Ensemble].encoders == [[1]]


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
