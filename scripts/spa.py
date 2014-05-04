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


