import nengo
import nengo.config


class Config(nengo.config.Config):
    def __init__(self, parent=None):
        super(Config, self).__init__(parent=parent)
        for klass in [nengo.Ensemble, nengo.Node]:
            self.configures(klass)
            self[klass].set_param('pos', nengo.config.Parameter(None))

        self.configures(nengo.Network)
        self[nengo.Network].set_param('scale', nengo.config.Parameter(1.0))
        self[nengo.Network].set_param('offset', nengo.config.Parameter((0,0)))

