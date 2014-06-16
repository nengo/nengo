import nengo

model = nengo.Network(label='critter')
with model:
    velocity = nengo.Node([0, 0], label='velocity')
    
    motor = nengo.Ensemble(200, dimensions=2, label='motor')
    
    nengo.Probe(motor)

    # make the position memory
    position = nengo.Ensemble(500, dimensions=2, label='position',
                              radius=3)
    nengo.Connection(position, position, synapse=0.1)
    nengo.Connection(motor, position, transform=0.1)
    
    nengo.Probe(position)
    
    
    # figure out which way is home
    home_location = [0.5, 0.5]
    
    home_dir = nengo.Ensemble(200, dimensions=2, label='home_dir')
    import numpy
    def compute_home(x):
        return (home_location - x) * 10
        
        #dx = home_location - x
        #norm = numpy.linalg.norm(dx)
        #if norm < 0.2:
        #    return [0, 0]
        #else:
        #    return dx / norm
    nengo.Connection(position, home_dir, function=compute_home) 
    nengo.Probe(home_dir)
    
    mode = nengo.Node(1, label='mode')
    
    d_velocity = nengo.Ensemble(300, dimensions=3, label='d_velocity',
                                radius=2)
    nengo.Connection(velocity, d_velocity[[0,1]])
    nengo.Connection(mode, d_velocity[2])
    def velocity_func(x):
        a, b, mode = x
        if mode > 0.5:
            return a, b
        else:
            return 0, 0
            
    nengo.Connection(d_velocity, motor, function=velocity_func)
            
    d_home = nengo.Ensemble(300, dimensions=3, label='d_home',
                                radius=2)
    nengo.Connection(home_dir, d_home[[0,1]])
    nengo.Connection(mode, d_home[2])
    def home_func(x):
        a, b, mode = x
        if mode < -0.5:
            return a, b
        else:
            return 0, 0
            
    nengo.Connection(d_home, motor, function=home_func)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    



import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 0.9033700098311324
gui[model].offset = -20.94864149010641,192.9167053429225
gui[motor].pos = 278.968, 44.328
gui[motor].scale = 1.000
gui[position].pos = 382.447, 53.794
gui[position].scale = 1.000
gui[home_dir].pos = 399.865, 151.456
gui[home_dir].scale = 1.000
gui[d_velocity].pos = 175.000, 38.930
gui[d_velocity].scale = 1.000
gui[d_home].pos = 193.818, 136.070
gui[d_home].scale = 1.000
gui[velocity].pos = 50.000, 50.000
gui[velocity].scale = 1.000
gui[mode].pos = 50.000, 125.000
gui[mode].scale = 1.000
