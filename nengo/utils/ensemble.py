from typing import Optional, Sequence, Tuple

import numpy as np

from ..ensemble import Ensemble
from ..npext import meshgrid_nd, norm
from ..types import SimulatorProtocol


def tuning_curves(
    ens: Ensemble,
    sim: SimulatorProtocol,
    inputs: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the tuning curves of an ensemble.

    That is the neuron responses in dependence of the vector represented by the
    ensemble.

    For 1-dimensional ensembles, the unpacked return value of this function
    can be passed directly to ``matplotlib.pyplot.plot``.

    Parameters
    ----------
    ens : nengo.Ensemble
        Ensemble to calculate the tuning curves of.
    sim : nengo.Simulator
        Simulator providing information about the built ensemble. (An unbuilt
        ensemble does not have tuning curves assigned to it.)
    inputs : sequence of ndarray, optional
        The inputs at which the tuning curves will be evaluated. For each of
        the ``D`` ensemble dimensions one array of dimensionality ``D`` is needed.
        The output of :func:`numpy.meshgrid` with ``indexing='ij'`` is in the
        right format.

    Returns
    -------
    inputs : ndarray
        The passed or auto-generated ``inputs``.
    activities : ndarray
        The activities of the individual neurons given the ``inputs``.
        For ensembles with 1 dimension, the rows correspond to the ``inputs``
        and the columns to individual neurons.
        For ensembles with > 1 dimension, the last dimension enumerates the
        neurons, the remaining dimensions map to ``inputs``.

    See Also
    --------
    response_curves
    """
    # note: imported here to avoid circular imports
    from nengo.builder.ensemble import (  # pylint: disable=import-outside-toplevel
        get_activities,
    )

    if inputs is None:
        inputs = np.linspace(-ens.radius, ens.radius)
        if ens.dimensions > 1:
            inputs = meshgrid_nd(*(ens.dimensions * [inputs]))
        else:
            inputs = [inputs]
        inputs = np.asarray(inputs).T
    else:
        inputs = np.asarray(inputs)

    eval_points = inputs.reshape((-1, ens.dimensions))
    activities = get_activities(sim.data[ens], ens, eval_points)
    return inputs, activities.reshape(inputs.shape[:-1] + (-1,))


def response_curves(
    ens: Ensemble,
    sim: SimulatorProtocol,
    inputs: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the response curves of an ensemble.

    That is the neuron responses in dependence of an already encoded value.
    This corresponds to the tuning curves along the neuron's preferred
    directions.

    Parameters
    ----------
    ens : nengo.Ensemble
        Ensemble to calculate the response curves of.
    sim : nengo.Simulator
        Simulator providing information about the built ensemble. (An unbuilt
        ensemble does not have response curves assigned to it.)
    inputs : 1d array, optional
        The inputs between -1 and 1 at which the neuron responses will be
        evaluated. They are assumed to be along each neuron's preferred
        direction.

    Returns
    -------
    inputs : 1d array
        The passed or auto-generated ``inputs``.
    activities : 2d array
        The activities of the individual neurons given the ``inputs``. The rows
        map to ``inputs`` and the columns to the neurons in the ensemble.

    See Also
    --------
    tuning_curves
    """

    if inputs is None:
        inputs = np.linspace(-1.0, 1.0)
    return inputs, ens.neuron_type.rates(inputs, sim.data[ens].gain, sim.data[ens].bias)


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
    i = index % cols  # find the 2d location of the indexth element
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


def sorted_neurons(ensemble, sim, iterations=100, seed=None):
    """Sort neurons in an ensemble by encoder and intercept.

    Parameters
    ----------
    ensemble : Ensemble
        The population of neurons to be sorted.
    sim : Simulator
        Simulator providing information about the built ensemble.
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

    .. testcode::

       from nengo.utils.ensemble import sorted_neurons
       from nengo.utils.matplotlib import rasterplot

       with nengo.Network() as net:
           ens = nengo.Ensemble(10, 1)
           ens_p = nengo.Probe(ens.neurons)

       with nengo.Simulator(net) as sim:
           sim.run(0.1)

           indices = sorted_neurons(ens, sim)
           rasterplot(sim.trange(), sim.data[ens_p][:, indices])

    .. testoutput::
       :hide:

       ...

    Notes
    -----
    The algorithm is for each encoder in the initial set, randomly
    pick another encoder and check to see if swapping those two
    encoders would reduce the average difference between the
    encoders and their neighbours.  Difference is measured as the
    dot product.  Each encoder has four neighbours (N, S, E, W),
    except for the ones on the edges which have fewer (no wrapping).
    This algorithm is repeated ``iterations`` times, so a total of
    ``iterations*N`` swaps are considered.
    """

    # Normalize all the encoders
    encoders = np.array(sim.data[ensemble].encoders)
    encoders /= norm(encoders, axis=1, keepdims=True)

    # Make an array with the starting order of the neurons
    N = encoders.shape[0]
    indices = np.arange(N)
    rng = np.random.RandomState(seed)

    for _ in range(iterations):
        target = rng.randint(0, N, N)  # pick random swap targets
        for i in range(N):
            j = target[i]
            if i != j:  # if not swapping with yourself
                # compute similarity score how we are (unswapped)
                sim1 = _similarity(encoders, i, N) + _similarity(encoders, j, N)
                # swap the encoder
                encoders[[i, j], :] = encoders[[j, i], :]
                indices[[i, j]] = indices[[j, i]]
                # compute similarity score how we are (swapped)
                sim2 = _similarity(encoders, i, N) + _similarity(encoders, j, N)

                # if we were better unswapped
                if sim1 > sim2:
                    # swap them back
                    encoders[[i, j], :] = encoders[[j, i], :]
                    indices[[i, j]] = indices[[j, i]]

    return indices
