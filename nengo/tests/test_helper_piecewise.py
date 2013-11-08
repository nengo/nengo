import numpy as np
from nengo.tests.helpers import unittest
import nengo

from nengo.helpers import piecewise

class TestPiecewise(unittest.TestCase):
    def test_basic(self):
        f = piecewise({0.5:1, 1.0:0})
        self.assertEqual(f(-10), [0])
        self.assertEqual(f(0), [0])
        self.assertEqual(f(0.25), [0])
        self.assertEqual(f(0.5), [1])
        self.assertEqual(f(0.75), [1])
        self.assertEqual(f(1.0), [0])
        self.assertEqual(f(1.5), [0])
        self.assertEqual(f(100), [0])

    def test_lists(self):
        f = piecewise({0.5:[1,0], 1.0:[0,1]})
        self.assertEqual(f(-10), [0,0])
        self.assertEqual(f(0), [0,0])
        self.assertEqual(f(0.25), [0,0])
        self.assertEqual(f(0.5), [1,0])
        self.assertEqual(f(0.75), [1,0])
        self.assertEqual(f(1.0), [0,1])
        self.assertEqual(f(1.5), [0,1])
        self.assertEqual(f(100), [0,1])


    def test_invalid_key(self):
        with self.assertRaises(TypeError):
            f = piecewise({0.5:1, 1:0, 'a':0.2})

    def test_invalid_length(self):
        with self.assertRaises(Exception):
            f = piecewise({0.5:[1,0], 1.0:[1,0,0]})

    def test_invalid_function_length(self):
        with self.assertRaises(Exception):
            f = piecewise({0.5:0, 1.0:lambda t: [t, t**2]})


    def test_function(self):
        f = piecewise({0:np.sin, 0.5:np.cos})
        self.assertEqual(f(0), [np.sin(0)])
        self.assertEqual(f(0.25), [np.sin(0.25)])
        self.assertEqual(f(0.4999), [np.sin(0.4999)])
        self.assertEqual(f(0.5), [np.cos(0.5)])
        self.assertEqual(f(0.75), [np.cos(0.75)])
        self.assertEqual(f(1.0), [np.cos(1.0)])

    def test_function_list(self):
        def func1(t):
            return t, t**2, t**3
        def func2(t):
            return t**4, t**5, t**6

        f = piecewise({0:func1, 0.5: func2})
        self.assertEqual(f(0), func1(0))
        self.assertEqual(f(0.25), func1(0.25))
        self.assertEqual(f(0.4999), func1(0.4999))
        self.assertEqual(f(0.5), func2(0.5))
        self.assertEqual(f(0.75), func2(0.75))
        self.assertEqual(f(1.0), func2(1.0))


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
