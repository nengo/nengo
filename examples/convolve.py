import nef.nef_theano as nef
import nef.convolution
import hrr

D=10

vocab=hrr.Vocabulary(D, include_pairs=True)
vocab.parse('a+b+c+d+e')

net=nef.Network('Convolution') #Create the network object

net.make('A',300,D) #Make a population of 300 neurons and 10 dimensions
net.make('B',300,D)
net.make('C',300,D)

conv=nef.convolution.make_convolution(net,'*','A','B','C',100) 
                #Call the code to construct a convolution network using 
                #the created populations and 100 neurons per dimension

net.run(1) # run for 1 second
