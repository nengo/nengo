import collections
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy as np


def tuning_curves(sim_ens, data):
    eval_points = np.array(sim_ens.eval_points)
    eval_points.sort(axis=0)
    activities = sim_ens.neurons.rates(
        eval_points * data['encoders'][sim_ens].T,
        data['gain'][sim_ens.neurons],
        data['bias'][sim_ens.neurons])
    return eval_points, activities


def piecewise(data):
    """Create a piecewise constant function from a dictionary.

    Given an input of data={0: 0, 0.5: 1, 0.75: -1, 1: 0} this will generate a
    function that returns 0 up until t=0.5, then outputs a 1 until t=0.75,
    then a -1 until t=1, and then returns 0 after that. This is meant as a
    shortcut for::

        def function(t):
            if t < 0.5:
                return 0
            elif t < 0.75
                return 1
            elif t < 1:
                return -1
            else:
                return 0

    The keys in the dictionary must be times (floats or ints). The values in
    the data dictionary can be floats, lists, or functions that return
    floats or lists. All lists must be of the same length.

    For times before the first specified time, it will default to zero (of
    the correct length). This means the above example can be simplified to::

        piecewise({0.5: 1, 0.75: -1, 1: 0})

    Parameters
    ----------
    data : dict
        The values to change to. Keys are the beginning time for the value.
        Values can be int, float, list, or functions that return those.

    Returns
    -------
    function:
        A function that takes a variable t and returns the corresponding
        value from the dictionary.

    Examples
    --------

      >>> func = piecewise({0.5: 1, 0.75: -1, 1: 0})
      >>> func(0.2)
      [0]
      >>> func(0.58)
      [1]

      >>> func = piecewise({0.5: [1, 0], 0.75: [0, 1]})
      >>> func(0.2)
      [0,0]
      >>> func(0.58)
      [1,0]
      >>> func(100)
      [0,1]

      >>> import math
      >>> func = piecewise({0: math.sin, 0.5: math.cos})
      >>> func(0.499)
      [0.47854771647582706]
      >>> func(0.5)
      [0.8775825618903728]

    """

    # first, sort the data (to simplify finding the right element
    # when calling the function)
    output_length = None  # the dimensionality of the returned values
    for time in data.keys():
        if not isinstance(time, (float, int)):
            raise TypeError('Keys must be times (floats or ints), not "%s"'
                            % repr(time.__class__))

        # figure out the length of this item
        if isinstance(data[time], collections.Callable):
            length = np.asarray(data[time](0.0)).size
        else:
            data[time] = np.asarray(data[time])
            length = data[time].size

        # make sure this is the same length as previous items
        if length != output_length and output_length is not None:
            raise ValueError('Invalid data for piecewise function: '
                             'time %4g has %d items instead of %d' %
                             (time, length, output_length))
        output_length = length

    # make a default output of 0 when t before what was passed
    data[np.finfo(float).min] = np.zeros(output_length)
    ordered_data = OrderedDict(sorted(data.items()))

    # build the function to return
    def piecewise_function(t, data=ordered_data):
        # get the t we'll use for output
        for time in (time for time in data.keys() if time <= t):
            out_t = time

        # if it's a function, call it
        if isinstance(data[out_t], collections.Callable):
            return np.asarray(data[out_t](t))
        return data[out_t]
    return piecewise_function


def _similarity(encoders, index, rows, cols=1):
    """Helper function to compute similarity for one encoder.

    Parameters
    ----------

    encoders: ndarray
        The encoders.
    index: int
        The encoder to compute for.
    rows: int
        The width of the 2d grid.
    cols: int
        The height of the 2d grid.
    """
    i = index % cols   # find the 2d location of the indexth element
    j = index // cols

    sim = 0  # total of dot products
    count = 0  # number of neighbours
    if i > 0:  # if we're not at the left edge, do the WEST comparison
        sim += np.dot(encoders[j * cols + i], encoders[j * cols + i - 1])
        count += 1
    if i < cols - 1:  # if we're not at the right edge, do EAST
        sim += np.dot(encoders[j * cols + i], encoders[j * cols + i + 1])
        count += 1
    if j > 0:  # if we're not at the top edge, do NORTH
        sim += np.dot(encoders[j * cols + i], encoders[(j - 1) * cols + i])
        count += 1
    if j < rows - 1:  # if we're not at the bottom edge, do SOUTH
        sim += np.dot(encoders[j * cols + i], encoders[(j + 1) * cols + i])
        count += 1
    return sim / count


def sorted_neurons(ensemble, encoders, iterations=100, seed=None):
    '''Sort neurons in an ensemble by encoder and intercept.

    Parameters
    ----------
    ensemble: nengo.Ensemble
        The population of neurons to be sorted.
        The ensemble must have its encoders specified.

    iterations: int
        The number of times to iterate during the sort.

    seed: float
        A random number seed.

    Returns
    -------
    indices: ndarray
        An array with sorted indices into the neurons in the ensemble

    Examples
    --------

    You can use this to generate an array of sorted indices for plotting. This
    can be done after collecting the data. E.g.

    >>>indices = sorted_neurons(simulator, 'My neurons')
    >>>plt.figure()
    >>>rasterplot(sim.data['My neurons.spikes'][:,indices])

    Algorithm
    ---------

    The algorithm is for each encoder in the initial set, randomly
    pick another encoder and check to see if swapping those two
    encoders would reduce the average difference between the
    encoders and their neighbours.  Difference is measured as the
    dot product.  Each encoder has four neighbours (N, S, E, W),
    except for the ones on the edges which have fewer (no wrapping).
    This algorithm is repeated `iterations` times, so a total of
    `iterations*N` swaps are considered.
    '''

    # Normalize all the neurons
    for i in np.arange(encoders.shape[0]):
        encoders[i, :] = encoders[i, :] / np.linalg.norm(encoders[i, :])

    # Make an array with the starting order of the neurons
    N = encoders.shape[0]
    indices = np.arange(N)
    rng = np.random.RandomState(seed)

    for k in range(iterations):
        target = rng.randint(0, N, N)  # pick random swap targets
        for i in range(N):
            j = target[i]
            if i != j:  # if not swapping with yourself
                # compute similarity score how we are (unswapped)
                sim1 = (_similarity(encoders, i, N)
                        + _similarity(encoders, j, N))
                # swap the encoder
                encoders[[i, j], :] = encoders[[j, i], :]
                indices[[i, j]] = indices[[j, i]]
                # compute similarity score how we are (swapped)
                sim2 = (_similarity(encoders, i, N)
                        + _similarity(encoders, j, N))

                # if we were better unswapped
                if sim1 > sim2:
                    # swap them back
                    encoders[[i, j], :] = encoders[[j, i], :]
                    indices[[i, j]] = indices[[j, i]]

    return indices


def whitenoise(step, high, rms=0.5, seed=None, dimensions=None):
    """Generate white noise inputs

    Parameters
    ----------
    step : float
        The step size of different frequencies to generate

    high : float
        The highest frequency to generate (should be a multiple of step)

    rms : float
        The RMS power of the signal

    seed : int or None
        Random number seed

    dimensions : int or None
        The number of different random signals to generate.  The resulting
        function will return an array of length `dimensions` for every
        point in time.  If `dimensions` is None, the resulting function will
        just return a float for each point in time.

    Returns
    -------
    function:
        A function that takes a variable t and returns the value of the
        randomly generated signal.  This value is a float if `dimensions` is
        None; otherwise it is a list of length `dimensions`.
    """
    rng = np.random.RandomState(seed)

    if dimensions is not None:
        signals = [whitenoise(
            step, high, rms=rms, seed=rng.randint(0x7ffffff))
            for i in range(dimensions)]

        def whitenoise_function(t, signals=signals):
            return [signal(t) for signal in signals]
        return whitenoise_function

    N = int(float(high) / step)  # number of samples
    frequencies = np.arange(1, N + 1) * step * 2 * np.pi  # frequency of each
    amplitude = rng.uniform(0, 1, N)  # amplitude for each sample
    phase = rng.uniform(0, 2 * np.pi, N)  # phase of each sample

    # compute the rms of the signal
    rawRMS = np.sqrt(np.sum(amplitude ** 2) / 2)
    amplitude = amplitude * rms / rawRMS  # rescale

    # create a function that computes the bases and weights them by amplitude
    def whitenoise_function(t, f=frequencies, a=amplitude, p=phase):
        return np.dot(a, np.sin((f * t) + p))

    return whitenoise_function
