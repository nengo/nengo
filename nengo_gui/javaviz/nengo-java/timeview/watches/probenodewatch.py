from ca.nengo.model.nef import NEFEnsemble
from ca.nengo.model import SimulationMode

from timeview import components
from timeview.javaviz import ProbeNode
from timeview.watches import watchtemplate

class ProbeNodeWatch(watchtemplate.WatchTemplate):
    def check(self, obj):
        return isinstance(obj, ProbeNode) and hasattr(obj, 'spike_probe')

    def spikes(self, obj):
        length = len(obj.spike_probe._value)
        val = [0.0] * length
        for i in range(length):
            val[i] = obj.spike_probe._value[i]
            obj.spike_probe._value[i] = 0.0

        return val

    def views(self, obj):
        r = [(None, None, None)]
            # Note that the above tuple is to reset popup menu to main popup menu in item.py

        if obj.spike_probe is not None:
            r.append(('spike raster', components.SpikeRaster,
                        dict(func=self.spikes, label=obj.name)))
            r.append(('firing rate', components.Grid, dict(func=self.spikes, min=0, max=lambda self: 200 * self.view.dt, filter=True, label=obj.name, audio=True)))
        if hasattr(obj, 'encoders') and len(obj.encoders[0])==2:
            r.append(('preferred directions', components.PreferredDirection, dict(func=self.spikes, min=0, max=lambda self: 500 * self.view.dt, filter=True, label=obj.name)))

        return r

    def priority(self):
        return 1
