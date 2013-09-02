from . import core


class RawProbe(object):
    """A raw probe is a wrapper around `nengo.core.Probe`.

    This wrapper is necessary because `nengo.Model` expects
    the `nengo.core.Probe` object to be `Probe.probe`.

    """
    def __init__(self, signal, sample_every):
        self.probe = core.Probe(signal, sample_every)

    @property
    def sample_every(self):
        return self.probe.dt

    @property
    def sample_rate(self):
        return 1.0 / self.probe.dt

    def add_to_model(self, model):
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.probe)


class Probe(object):
    """A probe is a dummy object that only has an input signal and probe.

    It is used as a target for a connection so that probe logic can
    reuse connection logic.

    Parameters
    ==========
    probed : Nengo object
        The object being probed.

    """
    def __init__(self, name, n_in, sample_every):
        self.input_signal = core.Signal(n=n_in, name="Probe(" + name + ")")
        self.probe = core.Probe(self.input_signal, sample_every)

    @property
    def name(self):
        return self.input_signal.name

    @property
    def sample_every(self):
        return self.probe.dt

    @property
    def sample_rate(self):
        return 1.0 / self.probe.dt

    def add_to_model(self, model):
        model.add(self.input_signal)
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.input_signal)
        model.remove(self.probe)
