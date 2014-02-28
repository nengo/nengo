import nengo
from nengo.spa.module import Module
from nengo.spa.rules import Rules
from nengo.utils.compat import iteritems


class BasalGanglia(nengo.networks.BasalGanglia, Module):
    def __init__(self, rules, input_synapse=0.002):
        self.rules = Rules(rules)
        assert self.rules.count > 0, "No rules in 'rules'"
        self.input_synapse = input_synapse
        # Two bases, so super() won't help us here
        Module.__init__(self)
        nengo.networks.BasalGanglia.__init__(self, dimensions=self.rules.count)

    def on_add(self, spa):
        Module.on_add(self, spa)

        self.rules.process(spa)

        for input, transform in iteritems(self.rules.get_inputs()):
            nengo.Connection(input, self.input,
                             transform=transform,
                             synapse=self.input_synapse)
