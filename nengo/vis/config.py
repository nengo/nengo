import nengo
import nengo.config
import nengo.params

class Config(nengo.config.Config):
    """Re-uses the Nengo config object for keeping track of GUI element
     parameters"""
    def __init__(self):
        super(Config, self).__init__()
        for klass in [nengo.Ensemble, nengo.Node]:
            self.configures(klass)
            self[klass].set_param('pos', nengo.params.Parameter(None))
            self[klass].set_param('scale', nengo.params.Parameter(None))
            self[klass].set_param('size', nengo.params.Parameter((50,50)))

        self.configures(nengo.Network)
        self[nengo.Network].set_param('scale', nengo.params.Parameter(None))
        self[nengo.Network].set_param('offset', nengo.params.Parameter((0,0)))
        self[nengo.Network].set_param('pos', nengo.params.Parameter(None))
        self[nengo.Network].set_param('size', nengo.params.Parameter(None)) #width/height
