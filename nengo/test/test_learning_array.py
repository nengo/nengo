"""This is a test file to test learning with network arrays

    Test cases:
    1) pre.array_size=N, post.array_size=1, error.array_size=1
    2) pre.array_size=1, post.array_size=N, error.array_size=1
    3) pre.array_size=1, post.array_size=N, error.array_size=N
    4) pre.array_size=N, post.array_size=N, error.array_size=N
   
    and for each of these
    Test cases:
        a) pre.dim=1, post.dim=1, error.dim=1
        b) pre.dim=N, post.dim=1, error.dim=1
        c) pre.dim=1, post.dim=N, error.dim=1
        d) pre.dim=N, post.dim=1, error.dim=N
        e) pre.dim=1, post.dim=N, error.dim=N
        f) pre.dim=N, post.dim=N, error.dim=N
"""

import math
import time

import itertools
import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

neurons = 30  # number of neurons in all ensembles
N = 2 # number of dimensions for multi-dimensional ensembles

test_array_sizes = [[N,1,1], 
                    [1,N,1], 
                    [N,1,N], 
                    [1,N,N], 
                    [N,N,N] ]

test_dims =        [[1,1,1], 
                    [N,1,1], 
                    [1,N,1], 
                    [N,1,N], 
                    [1,N,N], 
                    [N,N,N] ]
index_a = 0
index_d = 1

# generate set of different array and dimension sizes to test
test_cases = []
for i in range(len(test_array_sizes)):
    test_cases.extend([i]*len(test_dims))
test_cases = np.array( [test_cases] + 
                       [range(0, len(test_dims)) * 
                       len(test_array_sizes)] ).T

for i in range(test_cases.shape[0]):
    test_case = test_cases[i]

    array_sizes = test_array_sizes[test_case[index_a]]
    dims = test_dims[test_case[index_d]]
    
    # make sure that error dims == post dims
    if array_sizes[1] * dims[1] != array_sizes[2] * dims[2]:
        continue # if not, skip it, it's an invalid test case
        
    print 
    print 'test_case', test_case
    print 'test_array_size', array_sizes
    print 'test_dim', dims
    print

    net = nef.Network('Learning Test')
    net.make_input('in', value=[0.8,-.5])

    timer = time.time()

    net.make('A', neurons=neurons, 
                  dimensions=dims[0], 
                  array_size=array_sizes[0])
    net.make('B', neurons=neurons, 
                  dimensions=dims[1],
                  array_size=array_sizes[1])
    net.make('error', neurons=neurons, 
                  dimensions=dims[2], 
                  array_size=array_sizes[2])

    print "Made populations:", time.time() - timer

    net.learn(pre='A', post='B', error='error')

    net.connect('in', 'A')
    net.connect('A', 'error')
    net.connect('B', 'error', weight=-1)

    t_final = 5
    dt_step = 0.01
    pstc = 0.03

    Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
    Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
    Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
    E1p = net.make_probe('error', dt_sample=dt_step, pstc=pstc)

    print "starting simulation"

    start_time = time.time()
    net.run(t_final)
    print 'Simulated in ', time.time() - start_time, 'seconds'

    plt.ioff(); plt.close()

    t = np.linspace(0, t_final, len(Ap.get_data()))

    plt.plot(t, Ap.get_data())
    plt.plot(t, Bp.get_data())
    plt.plot(t, E1p.get_data())
    plt.legend( ['A'] * array_sizes[0] * dims[0] + 
                ['B'] * array_sizes[1] * dims[1] + 
                ['error'] * array_sizes[2] * dims[2] )
    plt.title('Normal learning')
    plt.tight_layout()
    plt.show()
