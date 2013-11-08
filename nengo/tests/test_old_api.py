import logging
import os

import numpy as np

import nengo
import nengo.old_api as nef
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestOldAPI(SimulatorTestCase):
    pass


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
