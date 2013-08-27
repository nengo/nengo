from . import core

class RawProbe(object):
    def __init__(self, signal, dt_sample):
        self.probe = core.Probe(signal, dt_sample)

    def add_to_model(self, model):
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.probe)


class FilteredProbe(object):
    def __init__(self, signal, dt_sample, filter, dt=0.001):
        fcoef, tcoef = core.filter_coefs(pstc=filter, dt=dt)
        self.orig_signal = signal
        self.signal = core.Signal(signal.n, name='probe(%s)' % signal.name)
        self.filter = core.Filter(fcoef, self.signal, self.signal)
        self.transform = core.Transform(tcoef, self.orig_signal, self.signal)
        self.probe = core.Probe(self.signal, dt_sample)

    def add_to_model(self, model):
        model.add(self.signal)
        model.add(self.filter)
        model.add(self.transform)
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.signal)
        model.remove(self.filter)
        model.remove(self.transform)
        model.remove(self.probe)
