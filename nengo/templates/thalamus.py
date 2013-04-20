def make(net, name='Thalamus', neurons=50, dimensions=2, 
         inhib_scale=3, tau_inhib=.005):
    """This method makes a Thalamus, which here is a set of tonically active
    ensembles, outputting the decoded value 1.
    
    :param Network net:
    :param string name: 
    :param int neurons:
    :param int dimensions:
    :param float inhib_scale:
    :param float tau_scale:
    :returns SubNetwork:
    """
    
    thalamus = net.make(name=name, neurons=neurons, dimensions=dimensions, 
        max_rate=(100,300), intercept=(-1, 0), radius=1, encoders=[[1]])    

    # setup inhibitory scaling matrix
    inhib_scaling_matrix = [[0]*dimensions for i in range(dimensions)]

    for i in range(dimensions):
        inhib_scaling_matrix[i][i] = -inhib_scale

    # setup inhibitory matrix
    inhib_matrix = []

    for i in range(dimensions):
        inhib_matrix_part = [[inhib_scaling_matrix[i]] * neurons]
        inhib_matrix.append(inhib_matrix_part[0])

    thalamus.addTermination('bg_input', inhib_matrix, tau_inhib, False)

    def addOne(x):
        return [x[0]+1]            
    net.connect('Thalamus', None, func=addOne, origin_name='xBiased', create_projection=False)
