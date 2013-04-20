import numpy as np

from .. import nef_theano as nef

net = nef.Network('Cleanup',seed=3)

D = 128
M = 50000
N1 = 100
N2 = 50
index = 0

def make_vector():
    v = np.random.normal(size=(D,))
    
    norm = np.linalg.norm(v)
    v = v / norm
    return v

print 'making words...'
words = [make_vector() for i in range(M)]
words = np.array(words)
print '...done'

net.make_array('A', N1, D)
net.make_array('B', N1, D)

net.make_array('C', N2, M)#,intercept=(0.6,0.9))
print 'made'

net.connect('A', 'C', words.T, pstc=0.1)
net.connect('C', 'B', words, pstc=0.1)

net.make_input('input', words[index])
net.connect('input', 'A', pstc=0.1)

net.run(0.001)

import time
start = time.time()

for i in range(5000):
    #print i,net.ensemble['A'].origin['X'].value.get_value()
    print i, words[index, :4],
    print net.nodes['C'].accumulator[0.1].projected_value.get_value()[:4]

    net.run(0.001)
    print (time.time() - start) / (i + 1)
