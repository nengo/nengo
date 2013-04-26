from unittest import TestCase
import nengo.model

class TestModel(TestCase):

    # -- subclasses may over-ride this to test other Model implementations
    Model = nengo.model.Model

    def setUp(self):
        pass
    
    def test_basic(self):
        model = self.Model("Test Model")
        self.assertEqual(model.name, "Test Model")
        
    def test_metadata(self):
        model = self.Model("Test Model")
        model.metadata["Test"] = "Test data"

        self.assertEqual(model.metadata, {"Test": "Test data"})
    
    def test_properties(self):
        pass

    def test_backend_type(self):
        pass
