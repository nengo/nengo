import nose
import numpy as np
from nengo.base import Model
from nengo.simulator import Simulator
from nengo.templates.circularconv import CircularConvolution

def test_mini(D=4, neurons_per_product=3):
    # D is dimensionality of semantic pointers

    m = Model(.001)
    rng = np.random.RandomState(1234)

    A = m.signal(D, value=rng.randn(D))
    B = m.signal(D, value=rng.randn(D))

    m.filter(1.0, A, A)
    m.filter(1.0, B, B)

    CircularConvolution(m, A, B,
        neurons_per_product=neurons_per_product)

    sim = Simulator(m)
    sim.run_steps(10)

    raise nose.SkipTest()

