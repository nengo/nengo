import itertools
import threading

import numpy as np


def signals_allclose(  # noqa: C901
    t,
    targets,
    signals,
    atol=1e-8,
    rtol=1e-5,
    buf=0,
    delay=0,
    plt=None,
    labels=None,
    individual_results=False,
    allclose=np.allclose,
):
    """Ensure all signal elements are within tolerances.

    Allows for delay, removing the beginning of the signal, and plotting.

    Parameters
    ----------
    t : array_like (T,)
        Simulation time for the points in ``target`` and ``signals``.
    targets : array_like (T, 1) or (T, N)
        Reference signal or signals for error comparison.
    signals : array_like (T, N)
        Signals to be tested against the target signals.
    atol, rtol : float
        Absolute and relative tolerances.
    buf : float
        Length of time (in seconds) to remove from the beginnings of signals.
    delay : float
        Amount of delay (in seconds) to account for when doing comparisons.
    plt : matplotlib.pyplot or mock
        Pyplot interface for plotting the results, unless it's mocked out.
    labels : list of string, length N
        Labels of each signal to use when plotting.
    individual_results : bool
        If True, returns a separate ``allclose`` result for each signal.
    allclose : callable
        Function to compare two arrays for similarity.
    """
    t = np.asarray(t)
    dt = t[1] - t[0]
    assert t.ndim == 1
    assert np.allclose(np.diff(t), dt)  # always use default allclose here

    targets = np.asarray(targets)
    signals = np.asarray(signals)
    if targets.ndim == 1:
        targets = targets.reshape((-1, 1))
    if signals.ndim == 1:
        signals = signals.reshape((-1, 1))
    assert targets.ndim == 2 and signals.ndim == 2
    assert t.size == targets.shape[0]
    assert t.size == signals.shape[0]
    assert targets.shape[1] == 1 or targets.shape[1] == signals.shape[1]

    buf = int(np.round(buf / dt))
    delay = int(np.round(delay / dt))
    slice1 = slice(buf, len(t) - delay)
    slice2 = slice(buf + delay, None)

    if plt is not None:
        if labels is None:
            labels = [None] * len(signals)
        elif isinstance(labels, str):
            labels = [labels]

        colors = ["b", "g", "r", "c", "m", "y", "k"]

        def plot_target(ax, x, b=0, c="k"):
            bound = atol + rtol * np.abs(x)
            y = x - b
            ax.plot(t[slice2], y[slice1], c + ":")
            ax.plot(t[slice2], (y + bound)[slice1], c + "--")
            ax.plot(t[slice2], (y - bound)[slice1], c + "--")

        # signal plot
        ax = plt.subplot(2, 1, 1)
        for y, label in zip(signals.T, labels):
            ax.plot(t, y, label=label)

        if targets.shape[1] == 1:
            plot_target(ax, targets[:, 0], c="k")
        else:
            color_cycle = itertools.cycle(colors)
            for x in targets.T:
                plot_target(ax, x, c=next(color_cycle))

        ax.set_ylabel("signal")
        if labels[0] is not None:
            lgd = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            plt.bbox_extra_artists = (lgd,)

        ax = plt.subplot(2, 1, 2)
        if targets.shape[1] == 1:
            x = targets[:, 0]
            plot_target(ax, x, b=x, c="k")
            for y, label in zip(signals.T, labels):
                ax.plot(t[slice2], y[slice2] - x[slice1])
        else:
            color_cycle = itertools.cycle(colors)
            for x, y, label in zip(targets.T, signals.T, labels):
                c = next(color_cycle)
                plot_target(ax, x, b=x, c=c)
                ax.plot(t[slice2], y[slice2] - x[slice1], c, label=label)

        ax.set_xlabel("time")
        ax.set_ylabel("error")

    if individual_results:
        if targets.shape[1] == 1:
            return [
                allclose(y[slice2], targets[slice1, 0], atol=atol, rtol=rtol)
                for y in signals.T
            ]
        else:
            return [
                allclose(y[slice2], x[slice1], atol=atol, rtol=rtol)
                for x, y in zip(targets.T, signals.T)
            ]
    else:
        return allclose(signals[slice2, :], targets[slice1, :], atol=atol, rtol=rtol)


class ThreadedAssertion:
    """Performs assertions in parallel.

    Starts a number of threads, waits for each thread to execute some
    initialization code, and then executes assertions in each thread.

    To use this class, create a derived class that implements ``init_thread``
    to start each thread running and ``assert_thread`` to check that the thread
    has run successfully. ``finish_thread`` can be used for any cleanup/shutdown
    of the thread.
    """

    class AssertionWorker(threading.Thread):
        def __init__(self, parent, barriers, n):
            super().__init__()
            self.parent = parent
            self.barriers = barriers
            self.n = n
            self.exception = None

        def run(self):
            self.parent.init_thread(self)

            self.barriers[self.n].set()
            for barrier in self.barriers:
                barrier.wait()

            try:
                self.parent.assert_thread(self)
            except Exception as e:  # pylint: disable=broad-except
                self.exception = e
            finally:
                self.parent.finish_thread(self)

    def __init__(self, n_threads):
        self.barriers = [threading.Event() for _ in range(n_threads)]
        self.threads = [
            self.AssertionWorker(self, self.barriers, i) for i in range(n_threads)
        ]

    def run(self):
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()
            if t.exception is not None:
                raise t.exception

    def init_thread(self, worker):
        pass

    def assert_thread(self, worker):
        raise NotImplementedError()

    def finish_thread(self, worker):
        pass
