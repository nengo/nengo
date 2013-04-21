"""This test file is for checking the run time of the theano code."""

import time

import nef_theano as nef

# make a lot of populations
popN = [10, 100, 1000]
N = [10, 100] 
D = [1, 2, 50]
for i in range(len(popN)):

    iset = range(popN[i])
    iset2 = range(popN[i]); iset2.reverse()

    for j in range(len(N)):

        for k in range(len(D)): 

            net=nef.Network('Runtime Test')
            net.make_input('in', [1])

            for l in iset:
                net.make(str(l), N[j], D[k])#, quick=True)

            # connect them all up
            for p1, p2 in zip(iset, iset2):
                net.connect(str(p1), str(p2))

            start_time = time.time()
            net.run(1, dt=.0005)
            print popN[i], " ", N[j], " ", D[k], " runtime: ", time.time() - start_time, "seconds"
