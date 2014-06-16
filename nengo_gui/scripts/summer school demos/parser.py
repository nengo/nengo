import nengo
import nengo.spa as spa

model = spa.SPA(label="Parser")
with model:
    model.vision = spa.Buffer(dimensions=32)
    model.noun = spa.Memory(dimensions=32)
    model.verb = spa.Memory(dimensions=32)
    
    model.sentence = spa.Buffer(dimensions=32)
    
    actions = spa.Actions(
                'dot(vision, WRITE + READ + RUN) --> verb=vision * 0.5',
                'dot(vision, ONE + TWO + THREE) --> noun=vision * 0.5',
                )
    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg)
    
    cortical_action = spa.Actions(
        'sentence = verb * VERB + noun * NOUN',
        )
    model.cortical = spa.Cortical(cortical_action)
        
    
    nengo.Probe(model.vision.state.output)
    nengo.Probe(model.noun.state.output)
    nengo.Probe(model.verb.state.output)
    nengo.Probe(model.sentence.state.output)
    nengo.Probe(model.bg.input)
    nengo.Probe(model.thalamus.actions.output)

import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.18071493982562714
gui[model].offset = 50.3731913821411,159.80831360032943
gui[model.vision].pos = 108.993, 434.190
gui[model.vision].scale = 1.000
gui[model.vision].size = 410.000, 267.000
gui[model.vision.state].pos = 108.993, 434.190
gui[model.vision.state].scale = 1.000
gui[model.vision.state].size = 330.000, 155.000
gui[model.vision.state.ea_ensembles[0]].pos = 108.993, 396.690
gui[model.vision.state.ea_ensembles[0]].scale = 1.000
gui[model.vision.state.ea_ensembles[1]].pos = 108.993, 471.690
gui[model.vision.state.ea_ensembles[1]].scale = 1.000
gui[model.vision.state.input].pos = -16.007, 434.190
gui[model.vision.state.input].scale = 1.000
gui[model.vision.state.output].pos = 233.993, 434.190
gui[model.vision.state.output].scale = 1.000
gui[model.noun].pos = 1946.357, 403.261
gui[model.noun].scale = 1.000
gui[model.noun].size = 285.000, 342.000
gui[model.noun.state].pos = 1946.357, 403.261
gui[model.noun.state].scale = 1.000
gui[model.noun.state].size = 205.000, 230.000
gui[model.noun.state.ea_ensembles[0]].pos = 1883.857, 328.261
gui[model.noun.state.ea_ensembles[0]].scale = 1.000
gui[model.noun.state.ea_ensembles[1]].pos = 1883.857, 403.261
gui[model.noun.state.ea_ensembles[1]].scale = 1.000
gui[model.noun.state.input].pos = 1883.857, 478.261
gui[model.noun.state.input].scale = 1.000
gui[model.noun.state.output].pos = 2008.857, 403.261
gui[model.noun.state.output].scale = 1.000
gui[model.verb].pos = 2006.519, 961.067
gui[model.verb].scale = 1.000
gui[model.verb].size = 285.000, 342.000
gui[model.verb.state].pos = 2006.519, 961.067
gui[model.verb.state].scale = 1.000
gui[model.verb.state].size = 205.000, 230.000
gui[model.verb.state.ea_ensembles[0]].pos = 1944.019, 886.067
gui[model.verb.state.ea_ensembles[0]].scale = 1.000
gui[model.verb.state.ea_ensembles[1]].pos = 1944.019, 961.067
gui[model.verb.state.ea_ensembles[1]].scale = 1.000
gui[model.verb.state.input].pos = 1944.019, 1036.067
gui[model.verb.state.input].scale = 1.000
gui[model.verb.state.output].pos = 2069.019, 961.067
gui[model.verb.state.output].scale = 1.000
gui[model.sentence].pos = 2583.399, 632.362
gui[model.sentence].scale = 1.000
gui[model.sentence].size = 410.000, 267.000
gui[model.sentence.state].pos = 2583.399, 632.362
gui[model.sentence.state].scale = 1.000
gui[model.sentence.state].size = 330.000, 155.000
gui[model.sentence.state.ea_ensembles[0]].pos = 2583.399, 594.862
gui[model.sentence.state.ea_ensembles[0]].scale = 1.000
gui[model.sentence.state.ea_ensembles[1]].pos = 2583.399, 669.862
gui[model.sentence.state.ea_ensembles[1]].scale = 1.000
gui[model.sentence.state.input].pos = 2458.399, 632.362
gui[model.sentence.state.input].scale = 1.000
gui[model.sentence.state.output].pos = 2708.399, 632.362
gui[model.sentence.state.output].scale = 1.000
gui[model.thalamus.bg].pos = 1151.662, -226.529
gui[model.thalamus.bg].scale = 1.000
gui[model.thalamus.bg].size = 1454.234, 767.000
gui[model.bg.input].pos = 576.810, -264.029
gui[model.bg.input].scale = 1.000
gui[model.bg.output].pos = 1651.810, -226.529
gui[model.bg.output].scale = 1.000
gui[model.bg.bias].pos = 576.810, -189.029
gui[model.bg.bias].scale = 1.000
gui[model.bg.networks[0]].pos = 876.810, -476.529
gui[model.bg.networks[0]].scale = 1.000
gui[model.bg.networks[0]].size = 330.000, 155.000
gui[model.bg.networks[0].ea_ensembles[0]].pos = 876.810, -514.029
gui[model.bg.networks[0].ea_ensembles[0]].scale = 1.000
gui[model.bg.networks[0].ea_ensembles[1]].pos = 876.810, -439.029
gui[model.bg.networks[0].ea_ensembles[1]].scale = 1.000
gui[model.bg.networks[0].input].pos = 751.810, -476.529
gui[model.bg.networks[0].input].scale = 1.000
gui[model.bg.networks[0].output].pos = 1001.810, -514.029
gui[model.bg.networks[0].output].scale = 1.000
gui[model.bg.networks[0].func_str].pos = 1001.810, -439.029
gui[model.bg.networks[0].func_str].scale = 1.000
gui[model.bg.networks[1]].pos = 876.810, -226.529
gui[model.bg.networks[1]].scale = 1.000
gui[model.bg.networks[1]].size = 330.000, 155.000
gui[model.bg.networks[1].ea_ensembles[0]].pos = 876.810, -264.029
gui[model.bg.networks[1].ea_ensembles[0]].scale = 1.000
gui[model.bg.networks[1].ea_ensembles[1]].pos = 876.810, -189.029
gui[model.bg.networks[1].ea_ensembles[1]].scale = 1.000
gui[model.bg.networks[1].input].pos = 751.810, -226.529
gui[model.bg.networks[1].input].scale = 1.000
gui[model.bg.networks[1].output].pos = 1001.810, -264.029
gui[model.bg.networks[1].output].scale = 1.000
gui[model.bg.networks[1].func_str].pos = 1001.810, -189.029
gui[model.bg.networks[1].func_str].scale = 1.000
gui[model.bg.networks[2]].pos = 876.810, 23.471
gui[model.bg.networks[2]].scale = 1.000
gui[model.bg.networks[2]].size = 330.000, 155.000
gui[model.bg.networks[2].ea_ensembles[0]].pos = 876.810, -14.029
gui[model.bg.networks[2].ea_ensembles[0]].scale = 1.000
gui[model.bg.networks[2].ea_ensembles[1]].pos = 876.810, 60.971
gui[model.bg.networks[2].ea_ensembles[1]].scale = 1.000
gui[model.bg.networks[2].input].pos = 751.810, 23.471
gui[model.bg.networks[2].input].scale = 1.000
gui[model.bg.networks[2].output].pos = 1001.810, -14.029
gui[model.bg.networks[2].output].scale = 1.000
gui[model.bg.networks[2].func_stn].pos = 1001.810, 60.971
gui[model.bg.networks[2].func_stn].scale = 1.000
gui[model.bg.networks[3]].pos = 1351.810, -351.529
gui[model.bg.networks[3]].scale = 1.000
gui[model.bg.networks[3]].size = 330.000, 155.000
gui[model.bg.networks[3].ea_ensembles[0]].pos = 1351.810, -389.029
gui[model.bg.networks[3].ea_ensembles[0]].scale = 1.000
gui[model.bg.networks[3].ea_ensembles[1]].pos = 1351.810, -314.029
gui[model.bg.networks[3].ea_ensembles[1]].scale = 1.000
gui[model.bg.networks[3].input].pos = 1226.810, -351.529
gui[model.bg.networks[3].input].scale = 1.000
gui[model.bg.networks[3].output].pos = 1476.810, -389.029
gui[model.bg.networks[3].output].scale = 1.000
gui[model.bg.networks[3].func_gpi].pos = 1476.810, -314.029
gui[model.bg.networks[3].func_gpi].scale = 1.000
gui[model.bg.networks[4]].pos = 1351.810, -101.529
gui[model.bg.networks[4]].scale = 1.000
gui[model.bg.networks[4]].size = 330.000, 155.000
gui[model.bg.networks[4].ea_ensembles[0]].pos = 1351.810, -139.029
gui[model.bg.networks[4].ea_ensembles[0]].scale = 1.000
gui[model.bg.networks[4].ea_ensembles[1]].pos = 1351.810, -64.029
gui[model.bg.networks[4].ea_ensembles[1]].scale = 1.000
gui[model.bg.networks[4].input].pos = 1226.810, -101.529
gui[model.bg.networks[4].input].scale = 1.000
gui[model.bg.networks[4].output].pos = 1476.810, -139.029
gui[model.bg.networks[4].output].scale = 1.000
gui[model.bg.networks[4].func_gpe].pos = 1476.810, -64.029
gui[model.bg.networks[4].func_gpe].scale = 1.000
gui[model.thalamus].pos = 1235.672, 1417.045
gui[model.thalamus].scale = 1.000
gui[model.thalamus].size = 1062.016, 517.000
gui[model.thalamus.ensembles[0]].pos = 982.164, 1217.045
gui[model.thalamus.ensembles[0]].scale = 1.000
gui[model.thalamus.ensembles[1]].pos = 982.164, 1292.045
gui[model.thalamus.ensembles[1]].scale = 1.000
gui[model.thalamus.bias].pos = 744.664, 1417.045
gui[model.thalamus.bias].scale = 1.000
gui[model.thalamus.actions].pos = 982.164, 1492.045
gui[model.thalamus.actions].scale = 1.000
gui[model.thalamus.actions].size = 205.000, 230.000
gui[model.thalamus.actions.ea_ensembles[0]].pos = 919.664, 1417.045
gui[model.thalamus.actions.ea_ensembles[0]].scale = 1.000
gui[model.thalamus.actions.ea_ensembles[1]].pos = 919.664, 1492.045
gui[model.thalamus.actions.ea_ensembles[1]].scale = 1.000
gui[model.thalamus.actions.input].pos = 919.664, 1567.045
gui[model.thalamus.actions.input].scale = 1.000
gui[model.thalamus.actions.output].pos = 1044.664, 1492.045
gui[model.thalamus.actions.output].scale = 1.000
gui[model.thalamus.networks[1]].pos = 1394.664, 1292.045
gui[model.thalamus.networks[1]].scale = 1.000
gui[model.thalamus.networks[1]].size = 330.000, 155.000
gui[model.thalamus.networks[1].ea_ensembles[0]].pos = 1394.664, 1254.545
gui[model.thalamus.networks[1].ea_ensembles[0]].scale = 1.000
gui[model.thalamus.networks[1].ea_ensembles[1]].pos = 1394.664, 1329.545
gui[model.thalamus.networks[1].ea_ensembles[1]].scale = 1.000
gui[model.thalamus.networks[1].input].pos = 1269.664, 1292.045
gui[model.thalamus.networks[1].input].scale = 1.000
gui[model.thalamus.networks[1].output].pos = 1519.664, 1292.045
gui[model.thalamus.networks[1].output].scale = 1.000
gui[model.thalamus.networks[2]].pos = 1394.664, 1542.045
gui[model.thalamus.networks[2]].scale = 1.000
gui[model.thalamus.networks[2]].size = 330.000, 155.000
gui[model.thalamus.networks[2].ea_ensembles[0]].pos = 1394.664, 1504.545
gui[model.thalamus.networks[2].ea_ensembles[0]].scale = 1.000
gui[model.thalamus.networks[2].ea_ensembles[1]].pos = 1394.664, 1579.545
gui[model.thalamus.networks[2].ea_ensembles[1]].scale = 1.000
gui[model.thalamus.networks[2].input].pos = 1269.664, 1542.045
gui[model.thalamus.networks[2].input].scale = 1.000
gui[model.thalamus.networks[2].output].pos = 1519.664, 1542.045
gui[model.thalamus.networks[2].output].scale = 1.000
gui[model.cortical].pos = 275.000, 912.500
gui[model.cortical].scale = 1.000
gui[model.cortical].size = 80.000, 80.000
gui[model.cortical.bias].pos = 275.000, 912.500
gui[model.cortical.bias].scale = 1.000
