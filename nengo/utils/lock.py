import errno
import os
import os.path
import platform
import signal
import time
import weakref

from nengo.exceptions import TimeoutError


_locks = weakref.WeakSet()


def _prepend_sig_handler(sig, fn):
    current_handler = signal.getsignal(sig)
    if current_handler == signal.SIG_DFL:
        def prepended_handler(sig, frame):
            fn(sig, frame)
            h = signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)
            signal.signal(sig, h)
    elif current_handler is None or current_handler == signal.SIG_IGN:
        prepended_handler = fn
    else:
        def prepended_handler(sig, frame):
            fn(sig, frame)
            current_handler(sig, frame)
    signal.signal(sig, prepended_handler)


def _release_locks_handler(sig, frame):
    for lock in _locks:
        lock.release()


# Build up a list of signals that terminate the process. The list of available
# signals will slightly differ depending on the platform (i.e. Windows does
# not support some signals). We do not include SIGINT which is translated to
# an exception by Python by default and SIGKILL for which no signal handler
# can be registered. This also assumes that these signals are not "mis-used"
# for things that will not terminate the process (because in that case we
# did not want to release the file locks). Because SIGUSR1 and SIGUSR2 are
# commonly used as non-terminating signal, we only register the signal handler
# if they still use the default (terminating) handler.
_termination_signals = [
    signal.SIGILL,
    signal.SIGABRT,
    signal.SIGFPE,
    signal.SIGSEGV,
    signal.SIGTERM,

]
if platform.system() not in ('', 'Windows'):
    _termination_signals.append(signal.SIGHUP)
    _termination_signals.append(signal.SIGQUIT)
    for sig in (signal.SIGUSR1, signal.SIGUSR2):
        if signal.getsignal(sig) == signal.SIG_DFL:
            _termination_signals.append(sig)

for sig in _termination_signals:
    _prepend_sig_handler(sig, _release_locks_handler)


class FileLock(object):
    def __init__(self, filename, timeout=10., poll=0.1):
        self.filename = filename
        self.timeout = timeout
        self.poll = poll
        self._fd = None
        _locks.add(self)

    def acquire(self):
        start = time.time()
        while True:
            try:
                self._fd = os.open(
                    self.filename, os.O_CREAT | os.O_RDWR | os.O_EXCL)
                return
            except OSError as err:
                if err.errno not in (errno.EEXIST, errno.EACCES):
                    raise
                elif time.time() - start >= self.timeout:
                    raise TimeoutError(
                        "Could not acquire lock '{filename}'.".format(
                            filename=self.filename))
                else:
                    time.sleep(self.poll)

    def release(self):
        if self._fd is not None:
            os.close(self._fd)
            os.remove(self.filename)
            self._fd = None

    @property
    def acquired(self):
        return self._fd is not None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def __del__(self):
        self.release()
