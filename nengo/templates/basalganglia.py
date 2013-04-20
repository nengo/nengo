def make_basal_ganglia(net, name='Basal Ganglia', dimensions=1, neurons=100, 
                       tau_ampa=0.002, tau_gaba=0.008, output_weight=1, 
                       radius=1.5):
    """This function creates a subnetwork with a model of the basal ganglia
    based off the paper (Gurney, Prescott, & Redgrave, 2001)
    
    :param NetWork net:
    :param string name:
    :param int dimensions:
    :param int neurons:
    :param float tau_ampa:
    :param float tau_gaba:
    :param float output_weight:
    :param float radius:

    :returns SubNetwork:
    """

    netbg = net.make_subnetwork(name)

    #TODO: make direct mode, implement with 1 neuron in direct mode
    netbg.make('input', neurons=1, dimensions=dimensions, mode='direct')
    #TODO: make direct mode, implement with 1 neuron in direct mode
    netbg.make('output', neurons=1, dimensions=dimensions, mode='direct')

    # connection weights from (Gurney, Prescott, & Redgrave, 2001)
    mm=1; mp=1; me=1; mg=1
    ws=1; wt=1; wm=1; wg=1; wp=0.9; we=0.3
    e=0.2; ep=-0.25; ee=-0.2; eg=-0.2
    le=0.2; lg=0.2

    # create the necessary neural ensembles
    #TODO: implement decoder_sign and set=1 for this population
    netbg.make('StrD1', neurons=neurons, array_size=dimensions, 
        dimensions=1, intercept=(e,1), encoders=[[1]], radius=radius)
    #TODO: implement decoder_sign and set=1 for this population
    netbg.make('StrD2', neurons=neurons, array_size=dimensions,
        dimensions=1, intercept=(e,1), encoders=[[1]], radius=radius)
    
    #TODO: implement decoder_sign and set=1 for this population
    netbg.make('STN', neurons=neurons, array_size=dimensions,
        dimensions=1, intercept=(ep,1), encoders=[[1]], radius=radius)
    #TODO: implement decoder_sign and set=1 for this population
    netbg.make('GPi', neurons=neurons, array_size=dimensions,
        dimensions=1, intercept=(eg,1), encoders=[[1]], radius=radius)
    #TODO: implement decoder_sign and set=1 for this population
    netbg.make('GPe', neurons=neurons, array_size=dimensions,
        dimensions=1, intercept=(ee,1), encoders=[[1]], radius=radius)

    # connect the input to the striatum and STN (excitatory)
    netbg.connect('input', 'StrD1', weight=ws*(1+lg), pstc=tau_ampa)
    netbg.connect('input', 'StrD2', weight=ws*(1-le), pstc=tau_ampa)
    netbg.connect('input', 'STN', weight=wt, pstc=tau_ampa)

    # connect the striatum to the GPi and GPe (inhibitory)
    def func_str(x):
        if x[0] < e: return 0
        return mm * (x[0] - e)
    netbg.connect('StrD1', 'GPi', func=func_str, weight=-wm, pstc=tau_gaba)
    netbg.connect('StrD2', 'GPe', func=func_str, weight=-wm, pstc=tau_gaba)

    # connect the STN to GPi and GPe (broad and excitatory)
    def func_stn(x):
        if x[0] < ep: return 0
        return mp * (x[0] - ep)
    tr = [[wp] * dimensions for i in range(dimensions)]    
    netbg.connect('STN', 'GPi', func=func_stn, transform=tr, pstc=tau_ampa)
    netbg.connect('STN', 'GPe', func=func_stn, transform=tr, pstc=tau_ampa)        

    # connect the GPe to GPi and STN (inhibitory)
    def func_gpe(x):
        if x[0] < ee: return 0
        return me * (x[0] - ee)
    netbg.connect('GPe', 'GPi', func=func_gpe, weight=-we, pstc=tau_gaba)
    netbg.connect('GPe', 'STN', func=func_gpe, weight=-wg, pstc=tau_gaba)

    #connect GPi to output (inhibitory)
    def func_gpi(x):
        if x[0]<eg: return 0
        return mg*(x[0]-eg)
    netbg.connect('GPi', 'output', func=func_gpi, pstc=tau_gaba, 
        weight=output_weight)
