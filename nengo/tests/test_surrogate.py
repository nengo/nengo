'''
Created on Dec 29, 2013

@author: bptripp
'''
import unittest
import nengo
import numpy as np

class TestSurrogateConnections(unittest.TestCase):

    def _run_connection(self, surrogate, transform1, transform2, function2=None):
        dA = transform1.shape[0]
        dB = transform2.shape[0]
        assert transform1.shape[1] == 1
        assert transform2.shape[1] == transform1.shape[0]
        
        m = nengo.Model("m", seed=123)    
        with m:
            input = nengo.Node(output = lambda t: t)
            max_rates=nengo.objects.Uniform(70, 130)
            n = 200
            
            if surrogate:
                neuronsA = nengo.LIFSurrogate(n)
                neuronsB = nengo.LIFSurrogate(n)
            else:
                neuronsA = nengo.LIF(n)
                neuronsB = nengo.LIF(n)
                
            A = nengo.Ensemble(neuronsA, dA, max_rates=max_rates)
            B = nengo.Ensemble(neuronsB, dB, max_rates=max_rates)
            nengo.Connection(input, A, transform=transform1)
            if function2 == None:
                nengo.Connection(A, B, transform=transform2)
            else:
                nengo.Connection(A, B, function=function2)
                  
            B_p = nengo.Probe(B, 'decoded_output', filter=0.01)
            sim = nengo.Simulator(m, dt=0.001)
            sim.run(2)
            return sim.data(B_p)
    
    def test_transform_1to1(self):
        transform1 = np.eye(1)
        transform2 = np.eye(1)
        lif_output = self._run_connection(False, transform1, transform2)
        surrogate_output = self._run_connection(True, transform1, transform2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.2, rtol=.01))
                    
    def test_transform_2to1(self):
        transform1 = np.array([[1], [1]])
        transform2 = np.array([[1, 0]])
        lif_output = self._run_connection(False, transform1, transform2)
        surrogate_output = self._run_connection(True, transform1, transform2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.2, rtol=.01))

    def test_transform_2to2(self):
        transform1 = np.array([[1], [1]])
        transform2 = np.eye(2)
        lif_output = self._run_connection(False, transform1, transform2)
        surrogate_output = self._run_connection(True, transform1, transform2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.3, rtol=.01))

    def test_transform_1to2(self):
        transform1 = np.eye(1)
        transform2 = np.array([[1], [1]])
        lif_output = self._run_connection(False, transform1, transform2)
        surrogate_output = self._run_connection(True, transform1, transform2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.3, rtol=.01))
    
    def test_transform_5to5(self):
        transform1 = np.array([[1], [0], [0], [0], [0]])
        transform2 = np.eye(5)
        lif_output = self._run_connection(False, transform1, transform2)
        surrogate_output = self._run_connection(True, transform1, transform2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.4, rtol=.01))
    
    def test_function_1to1(self):
        transform1 = np.eye(1)
        transform2 = np.array([[0]])  
        function2 = lambda x: np.sin(x)
        lif_output = self._run_connection(False, transform1, transform2, function2=function2)
        surrogate_output = self._run_connection(True, transform1, transform2, function2=function2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.2, rtol=.01))
    
    def test_function_2to1(self):
        transform1 = np.array([[1], [1]])
        transform2 = np.array([[0, 0]])
        function2 = lambda x: np.sin(x[0])*np.square(x[1])
        lif_output = self._run_connection(False, transform1, transform2, function2=function2)
        surrogate_output = self._run_connection(True, transform1, transform2, function2=function2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.2, rtol=.01))

    def test_function_2to2(self):
        transform1 = np.array([[1], [1]])
        transform2 = np.zeros([2,2])
        function2 = lambda x: [np.sin(x[0])*np.square(x[1]), x[0]]
        lif_output = self._run_connection(False, transform1, transform2, function2=function2)
        surrogate_output = self._run_connection(True, transform1, transform2, function2=function2)
        self.assertTrue(np.allclose(lif_output, surrogate_output, atol=.2, rtol=.01))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()