import nengo


class Oscillator(nengo.Network):
    """An ensemble that connects to itself in a manner similar to the 
    integrator, however here the two dimensions interact with each other."""
    def __init__(self, recurrent_tau, frequency, **ens_args):
        self.oscillator_input = nengo.Node(label="In", size_in=2)
        self.ensemble = nengo.Ensemble(
            label="Oscillator", dimensions=2, **ens_args)
        tA = [[1, -frequency * recurrent_tau],
              [frequency * recurrent_tau, 1]]
        nengo.Connection(self.ensemble, self.ensemble,
                         synapse=recurrent_tau, transform=tA)
        nengo.Connection(self.oscillator_input, self.ensemble)
