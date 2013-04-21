from unittest import TestCase
from .. import nef as nef
from nef.model import Model
from nef.ensemble import Ensemble
from nef.neuron.lif import LIFNeuron

class TestEnsemble(TestCase):
    def setUp(self):
        self.model = Model("Test Model")
        self.ens1 = self.model.make_ensemble("ens1", 100, 1, (100,200), (-1,1), 1, None, LIFNeuron())


    def test_basic(self):
        assertEqual(self.ens1, self.model.get("ens1"))
        assertEqual(self.ens1.name, "ens1")
        assertEqual(self.ens1.neurons, 100)
        assertEqual(self.ens1.dimensions, 1)
        assertEqual(self.ens1.max_rate, (100,200))
        assertEqual(self.ens1.intercept, (-1,1))
        assertEqual(self.ens1.radius, 1)
        assertEqual(self.ens1.encoders, None)


    def test_advanced(self):
        #self.ens1.rates = [...]
        #self.ens1.intercepts = [...]
        # And more stuff
        pass

  
    def test_run(self):
        data = []

        self.model.probe('ens1', 1, False)
        self.model.run(1, dt = 0.001, output = data)

        assert True #MSE(data - ideal) < threshold


    def test_run_rate(self):
        data = []
        
        self.model.setMode('rate') ## How do we do this?
        self.model.probe('ens1', 1, False)
        self.model.run(1, dt = 0.001, output = data)

        assert True #MSE(data - ideal) < threshold 


    def test_run_direct(self):
        data = []
        
        self.model.setMode('direct') 
        self.model.probe('ens1', 1, False)
        self.model.run(1, dt = 0.001, output = data)

        assert True #MSE(data - ideal) < threshold 

    def test_ensemble(self):
        # Unit test for ensemble???
        pass