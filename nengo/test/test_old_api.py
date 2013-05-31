import numpy as np
from nengo.old_api import Network
from matplotlib import pyplot as plt

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def test_basic_1(show=False):
    """
    Create a network with sin(t) being represented by
    a population of spiking neurons. Assert that the
    decoded value from the population is close to the
    true value (which is input to the population).

    Expected duration of test: about .7 seconds
    """

    net = Network('Runtime Test', dt=0.001, seed=123)
    print 'make_input'
    net.make_input('in', value=np.sin)
    print 'make A'
    net.make('A', 1000, 1)
    print 'connecting in -> A'
    net.connect('in', 'A')
    A_fast_probe = net.make_probe('A', dt_sample=0.01, pstc=0.001)
    A_med_probe = net.make_probe('A', dt_sample=0.01, pstc=0.01)
    A_slow_probe = net.make_probe('A', dt_sample=0.01, pstc=0.1)
    in_probe = net.make_probe('in', dt_sample=0.01, pstc=0.01)

    net.run(1.0)

    target = np.sin(np.arange(0, 1000, 10) / 1000.)
    target.shape = (100, 1)

    assert np.allclose(target, in_probe.get_data())
    assert rmse(target, A_fast_probe.get_data()) < .25
    assert rmse(target, A_med_probe.get_data()) < .025
    assert rmse(target, A_slow_probe.get_data()) < 0.1

    for speed in 'fast', 'med', 'slow':
        probe = locals()['A_%s_probe' % speed]
        data = np.asarray(probe.get_data()).flatten()
        plt.plot(data, label=speed)

    in_data = np.asarray(in_probe.get_data()).flatten()

    plt.plot(in_data, label='in')
    plt.legend(loc='upper left')

    if show:
        plt.show()


