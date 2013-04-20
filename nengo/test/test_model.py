from unittest import TestCase
from nengo.nef.model import Model

class TestModel(TestCase):
    def setUp(self):
        pass
    
    def test_basic(self):
        model = Model("Test Model")
        assertEqual(model.name, "Test Model")
        
    def test_metadata(self):
        model = Model("Test Model")
        model.metadata["Test"] = "Test data"

        assertEqual(model.metadata, {"Test": "Test data"})
    
    def test_properties(self):
        pass

    def test_backend_type(self):
        pass