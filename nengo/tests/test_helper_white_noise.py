import numpy as np
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest
import nengo

from nengo.helpers import white_noise

class TestWhiteNoise(unittest.TestCase):
    def test_rms(self):
        t = np.linspace(0, 1, 1000)

        for rms_desired in [0, 0.5, 1, 100]:
            f = white_noise(1, 100, rms=rms_desired)

            rms = np.sqrt(np.mean([f(tt)**2 for tt in t]))

            self.assertTrue(np.allclose(rms, rms_desired, atol=.1, rtol=.01))

    def test_array(self):
        rms_desired = 0.5
        f = white_noise(1, 5, rms=rms_desired, dimensions=5)

        t = np.linspace(0, 1, 1000)
        data = np.array([f(tt) for tt in t])

        rms = np.sqrt(np.mean(data**2, axis=0))

        self.assertTrue(np.allclose(rms, rms_desired, atol=.1, rtol=.01))




if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
