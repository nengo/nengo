import nengo
import nengo.spa as spa

with spa.SPA() as model:
    model.vision = spa.Buffer(32)
    
    actions = spa.Actions(
        'dot(vision, A) --> vision=B',
        'dot(vision, B) --> vision=C',
        'dot(vision, C) --> vision=A',
        )        

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)




import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9096506079108221
gui[model].offset = -608016.6666607099,-5804180.301175613
gui[model.vision.state.ea_ensembles[0]].pos = 1121766.388, 6453724.880
gui[model.vision.state.ea_ensembles[0]].scale = 1.211
gui[model.vision.state.ea_ensembles[1]].pos = 1121835.121, 6454091.899
gui[model.vision.state.ea_ensembles[1]].scale = 1.211
gui[model.vision.state.input].pos = 1122188.935, 6453994.169
gui[model.vision.state.input].scale = 1.211
gui[model.vision.state.output].pos = 1121374.270, 6453846.648
gui[model.vision.state.output].scale = 1.211
gui[model.bg.input].pos = 736816.298, 5814161.926
gui[model.bg.input].scale = 0.280
gui[model.bg.output].pos = 735901.246, 5812294.113
gui[model.bg.output].scale = 0.280
gui[model.bg.bias].pos = 736194.240, 5813152.528
gui[model.bg.bias].scale = 0.280
gui[model.bg.networks[0].ea_ensembles[0]].pos = 889882.295, 6060888.151
gui[model.bg.networks[0].ea_ensembles[0]].scale = 0.280
gui[model.bg.networks[0].ea_ensembles[1]].pos = 889815.499, 6060803.058
gui[model.bg.networks[0].ea_ensembles[1]].scale = 0.280
gui[model.bg.networks[0].ea_ensembles[2]].pos = 889893.622, 6060812.680
gui[model.bg.networks[0].ea_ensembles[2]].scale = 0.280
gui[model.bg.networks[0].input].pos = 889793.793, 6060863.295
gui[model.bg.networks[0].input].scale = 0.280
gui[model.bg.networks[0].output].pos = 889841.395, 6060791.923
gui[model.bg.networks[0].output].scale = 0.280
gui[model.bg.networks[0].func_str].pos = 889828.411, 6060747.747
gui[model.bg.networks[0].func_str].scale = 0.280
gui[model.bg.networks[1].ea_ensembles[0]].pos = 889575.580, 6059799.731
gui[model.bg.networks[1].ea_ensembles[0]].scale = 0.277
gui[model.bg.networks[1].ea_ensembles[1]].pos = 889635.221, 6059570.802
gui[model.bg.networks[1].ea_ensembles[1]].scale = 0.277
gui[model.bg.networks[1].ea_ensembles[2]].pos = 889469.760, 6059682.967
gui[model.bg.networks[1].ea_ensembles[2]].scale = 0.277
gui[model.bg.networks[1].input].pos = 889466.652, 6059523.338
gui[model.bg.networks[1].input].scale = 0.277
gui[model.bg.networks[1].output].pos = 889568.035, 6059834.429
gui[model.bg.networks[1].output].scale = 0.277
gui[model.bg.networks[1].func_str].pos = 889584.878, 6059743.760
gui[model.bg.networks[1].func_str].scale = 0.277
gui[model.bg.networks[2].ea_ensembles[0]].pos = 918604.203, 6107446.613
gui[model.bg.networks[2].ea_ensembles[0]].scale = 0.280
gui[model.bg.networks[2].ea_ensembles[1]].pos = 918759.213, 6107638.182
gui[model.bg.networks[2].ea_ensembles[1]].scale = 0.280
gui[model.bg.networks[2].ea_ensembles[2]].pos = 918595.525, 6107517.312
gui[model.bg.networks[2].ea_ensembles[2]].scale = 0.280
gui[model.bg.networks[2].input].pos = 918539.057, 6107412.466
gui[model.bg.networks[2].input].scale = 0.280
gui[model.bg.networks[2].output].pos = 918662.823, 6107516.130
gui[model.bg.networks[2].output].scale = 0.280
gui[model.bg.networks[2].func_stn].pos = 918527.744, 6107513.284
gui[model.bg.networks[2].func_stn].scale = 0.280
gui[model.bg.networks[3].ea_ensembles[0]].pos = 960452.192, 6175296.289
gui[model.bg.networks[3].ea_ensembles[0]].scale = 0.280
gui[model.bg.networks[3].ea_ensembles[1]].pos = 960464.758, 6175271.070
gui[model.bg.networks[3].ea_ensembles[1]].scale = 0.280
gui[model.bg.networks[3].ea_ensembles[2]].pos = 960582.930, 6175158.511
gui[model.bg.networks[3].ea_ensembles[2]].scale = 0.280
gui[model.bg.networks[3].input].pos = 960630.722, 6175304.628
gui[model.bg.networks[3].input].scale = 0.280
gui[model.bg.networks[3].output].pos = 960357.416, 6174922.655
gui[model.bg.networks[3].output].scale = 0.280
gui[model.bg.networks[3].func_gpi].pos = 960578.210, 6175177.024
gui[model.bg.networks[3].func_gpi].scale = 0.280
gui[model.bg.networks[4].ea_ensembles[0]].pos = 791664.540, 6103344.463
gui[model.bg.networks[4].ea_ensembles[0]].scale = 0.280
gui[model.bg.networks[4].ea_ensembles[1]].pos = 792341.013, 6104373.589
gui[model.bg.networks[4].ea_ensembles[1]].scale = 0.280
gui[model.bg.networks[4].ea_ensembles[2]].pos = 791084.334, 6102480.334
gui[model.bg.networks[4].ea_ensembles[2]].scale = 0.280
gui[model.bg.networks[4].input].pos = 791630.272, 6103317.416
gui[model.bg.networks[4].input].scale = 0.280
gui[model.bg.networks[4].output].pos = 792328.732, 6104286.663
gui[model.bg.networks[4].output].scale = 0.280
gui[model.bg.networks[4].func_gpe].pos = 791080.020, 6102529.250
gui[model.bg.networks[4].func_gpe].scale = 0.280
gui[model.thal.bias].pos = 669134.536, 6380887.880
gui[model.thal.bias].scale = 0.141
gui[model.thal.actions.ea_ensembles[0]].pos = 669030.234, 6380885.384
gui[model.thal.actions.ea_ensembles[0]].scale = 0.099
gui[model.thal.actions.ea_ensembles[1]].pos = 668985.522, 6380914.355
gui[model.thal.actions.ea_ensembles[1]].scale = 0.099
gui[model.thal.actions.ea_ensembles[2]].pos = 668983.600, 6380885.650
gui[model.thal.actions.ea_ensembles[2]].scale = 0.099
gui[model.thal.actions.input].pos = 669020.056, 6380914.084
gui[model.thal.actions.input].scale = 0.099
gui[model.thal.actions.output].pos = 669003.769, 6380942.757
gui[model.thal.actions.output].scale = 0.099
