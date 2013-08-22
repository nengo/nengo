"""
Defines an SVG visualization algorithm for the low level Model class.
"""

import numpy as np

import StringIO
from IPython.display import display, SVG
from simulator import Simulator


POST_00 = 0 # initial
POST_EN = 1 # after encoders
POST_NL = 2 # after non-linearities
POST_DE = 3 # after decoders
POST_TF = 4 # after transforms
POST_FI = 5 # after filters


class SimulatorVisualizer(object):
    def __init__(self, sim_obj):
        self.sim_obj = sim_obj
        self.signals = signals = list(sim_obj.signals)

        usage = np.zeros((6, len(self.signals)))

        for enc in sim_obj.model.encoders:
            usage[POST_00, signals.index(enc.sig)] += 1
            usage[POST_EN, signals.index(enc.pop.input_signal)] += 1
            usage[POST_EN, signals.index(enc.pop.bias_signal)] += 1

        for nl in sim_obj.model.nonlinearities:
            usage[POST_EN, signals.index(nl.input_signal)] += 1
            usage[POST_NL, signals.index(nl.output_signal)] += 1

        for dec in sim_obj.model.decoders:
            usage[POST_NL, signals.index(dec.pop.output_signal)] += 1
            usage[POST_DE, signals.index(dec.sig)] += 1

        for tf in sim_obj.model.transforms:
            usage[POST_DE, signals.index(tf.insig.base)] += 1
            usage[POST_TF, signals.index(tf.outsig.base)] += 1

        for filt in sim_obj.model.filters:
            usage[POST_00, signals.index(filt.oldsig.base)] += 1
            usage[POST_TF, signals.index(filt.newsig.base)] += 1

        usage_sums = usage.sum(axis=0)

        self.usage = usage
        self.usage_sums = usage_sums

    def render_svg(self,
            x_spacing = 250,
            y_spacing = 100,
            y_offset = 50,
            x_offset = 10,
            ):
        sio = StringIO.StringIO()
        usage = self.usage
        usage_sums = self.usage_sums
        signals = self.signals
        sim_obj = self.sim_obj
        num_columns = np.sum(usage_sums > 0)
        active_lt = np.cumsum(usage_sums > 0) - (usage_sums[0] > 0)

        def append(msg):
            print >> sio, msg

        append('<svg xmlns="http://www.w3.org/2000/svg" version="1.1"'
            ' width="%s" height="%s">' % (
            num_columns * x_spacing, y_offset * 2 + len(usage) * y_spacing))

        def append_circle(x, y):
            append('<circle cx="%s" cy="%s" r="5" stroke="black"'
                    ' stroke-width="2" fill="red" />' % (x, y))

        # print headers
        i_offset = x_offset
        for i, s in enumerate(signals):
            if usage_sums[i] == 0:
               continue
            append('<text x="%s" y="%s">%s</text>' % (
                i_offset, y_offset, str(s).replace('<', '?')))
            i_offset += x_spacing

        # print block names
        append('<text x="%s" y="%s">Encoders</text>' % (
            x_offset,
            y_offset + 1.5 * y_spacing))
        append('<text x="%s" y="%s">Non-linearities</text>' % (
            x_offset,
            y_offset + 2.5 * y_spacing))
        append('<text x="%s" y="%s">Decoders</text>' % (
            x_offset,
            y_offset + 3.5 * y_spacing))
        append('<text x="%s" y="%s">Transforms & Filters</text>' % (
            x_offset,
            y_offset + 4.5 * y_spacing))

        # print input signal row
        i_offset = x_offset
        for i, s in enumerate(signals):
            if usage_sums[i] == 0:
               continue
            append_circle(i_offset, y_offset + 1 * y_spacing)
            i_offset += x_spacing

        # print the encoder output row
        i_offset = x_offset
        for i, s in enumerate(signals):
            if usage_sums[i] == 0:
               continue
            append_circle(i_offset, y_offset + 2 * y_spacing)
            i_offset += x_spacing

        for i, enc in enumerate(sim_obj.model.encoders):
            pos_sig = active_lt[signals.index(enc.sig)]
            pos_inp = active_lt[signals.index(enc.pop.input_signal)]
            pos_bias = active_lt[signals.index(enc.pop.bias_signal)]

            append('<line x1="%s" y1="%s" x2="%s" y2="%s"'
                    ' style="stroke:rgb(255,0,0);stroke-width:2"/>' % (
                pos_sig * x_spacing + x_offset,
                y_offset + 1 * y_spacing,
                pos_inp * x_spacing + x_offset,
                y_offset + 2 * y_spacing,
                ))
            append('<line x1="%s" y1="%s" x2="%s" y2="%s"'
                    ' style="stroke:rgb(255,0,0);stroke-width:2"/>' % (
                pos_bias * x_spacing + x_offset,
                y_offset + 1 * y_spacing,
                pos_inp * x_spacing + x_offset,
                y_offset + 2 * y_spacing,
                ))

        # print the non-linearities output row
        i_offset = x_offset
        for i, s in enumerate(signals):
            if usage_sums[i] == 0:
               continue
            append_circle(i_offset, y_offset + 3 * y_spacing)
            i_offset += x_spacing

        for nl in sim_obj.model.nonlinearities:
            pos_sig = active_lt[signals.index(nl.input_signal)]
            pos_inp = active_lt[signals.index(nl.output_signal)]
            append('<line x1="%s" y1="%s" x2="%s" y2="%s"'
                    'style="stroke:rgb(255,0,0);stroke-width:2"/>' % (
                pos_sig * x_spacing + x_offset,
                y_offset + 2 * y_spacing,
                pos_inp * x_spacing + x_offset,
                y_offset + 3 * y_spacing,
                ))

        # print the decoder output row
        i_offset = x_offset
        for i, s in enumerate(signals):
            if usage_sums[i] == 0:
               continue
            append_circle(i_offset, y_offset + 4 * y_spacing)
            i_offset += x_spacing

        for dec in sim_obj.model.decoders:
            src = active_lt[signals.index(dec.pop.output_signal)]
            dst = active_lt[signals.index(dec.sig)]
            append('<line x1="%s" y1="%s" x2="%s" y2="%s"'
                    ' style="stroke:rgb(255,0,0);stroke-width:2"/>' % (
                src * x_spacing + x_offset,
                y_offset + 3 * y_spacing,
                dst * x_spacing + x_offset,
                y_offset + 4 * y_spacing,
                ))

        # print the transform output row
        i_offset = x_offset
        for i, s in enumerate(signals):
            if usage_sums[i] == 0:
               continue
            append_circle(i_offset, y_offset + 5 * y_spacing)
            i_offset += x_spacing

        for tf in sim_obj.model.transforms:
            src = active_lt[signals.index(tf.insig.base)]
            dst = active_lt[signals.index(tf.outsig.base)]
            append('<line x1="%s" y1="%s" x2="%s" y2="%s"'
                    ' style="stroke:rgb(255,0,0);stroke-width:2"/>' % (
                src * x_spacing + x_offset,
                y_offset + 4 * y_spacing,
                dst * x_spacing + x_offset,
                y_offset + 5 * y_spacing,
                ))

        for filt in sim_obj.model.filters:
            src = active_lt[signals.index(filt.oldsig.base)]
            dst = active_lt[signals.index(filt.newsig.base)]
            append('<line x1="%s" y1="%s" x2="%s" y2="%s"'
                    ' style="stroke:rgb(0,0,0);stroke-width:2"/>' % (
                src * x_spacing + x_offset,
                y_offset + 1 * y_spacing,
                dst * x_spacing + x_offset,
                y_offset + 5 * y_spacing,
                ))

        append('</svg>')
        rval = sio.getvalue()
        return rval


def simulator_to_svg(sim_obj):
    vis = SimulatorVisualizer(sim_obj)
    # display(SVG(vis.render_svg()))
    return vis.render_svg()

