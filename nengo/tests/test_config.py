import pytest

import nengo
import nengo.config


def test_config():
    @nengo.config.configures(nengo.Ensemble)
    class TestConfigEnsemble(nengo.config.ConfigItem):
        something = nengo.config.Parameter(None)
        other = nengo.config.Parameter(0)

    @nengo.config.configures(nengo.Connection)
    class TestConfigConnection(nengo.config.ConfigItem):
        something_else = nengo.config.Parameter(None)

    class TestConfig(nengo.config.Config):
        config_items = [TestConfigEnsemble, TestConfigConnection]

    model = nengo.Model()
    with model:
        a = nengo.Ensemble(nengo.LIF(50), 1)
        b = nengo.Ensemble(nengo.LIF(90), 1)
        a2b = nengo.Connection(a, b, filter=0.01)

    config = TestConfig()

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

    model = nengo.Model()
    with model:
        a = nengo.Ensemble(50, 1)
        b = nengo.Ensemble(90, 1)

    class TestConfig(nengo.config.Config):
        config_items = [TestConfigEnsemble]
    config = TestConfig()

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

    with pytest.raises(AttributeError):
        class TestConfig(nengo.config.Config):
            pass
        TestConfig()
    with pytest.raises(AttributeError):
        class TestConfig(nengo.config.Config):
            config_items = [1, 2, 3]
        TestConfig()
    with pytest.raises(AttributeError):
        class TestConfig(nengo.config.Config):
            config_items = [TestBadConfigConnection]
        TestConfig()
    with pytest.raises(AttributeError):
        class TestConfig(nengo.config.Config):
            config_items = [TestConfigEnsemble, TestBadConfigConnection]
        TestConfig()

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
