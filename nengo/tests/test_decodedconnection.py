import logging

import numpy as np

import nengo
from nengo.objects import Ensemble
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest

logger = logging.getLogger(__name__)

class TestDecodedConnection(SimulatorTestCase):

    def test_basic(self):
        pass

    def test_modulatory(self):
        """Make sure modulatory connections don't impart current."""
        pass
