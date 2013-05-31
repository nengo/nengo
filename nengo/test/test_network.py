from unittest import TestCase
from nengo.model import Model
from nengo.nef.network import Network

class TestNetwork(TestCase):
    def setUp(self):
        model = Model("Test Model")
    
