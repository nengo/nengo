from . import objects

class RawProbe(object):
    def __init__(self, signal, dt_sample):
        self.probe = objects.Probe(signal, dt_sample)

    def add_to_model(self, model):
        model.add(self.probe)

    def remove_from_model(self, model):
        model.remove(self.probe)


class FilteredProbe(object):
    def __init__(self, signal, dt_sample, filter):
        fcoef, tcoef = objects.filter_coefs(pstc=filter, dt=dt)
        self.orig_signal = signal
        self.signal = objects.Signal(signal.n, name='probe(%s)' % signal.name)
        self.filter = objects.Filter(fcoef, self.signal, self.signal)
        self.transform = objects.Transform(tcoef, self.orig_signal, self.signal)
        self.probe = objects.Probe(self.signal, dt_sample)

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
