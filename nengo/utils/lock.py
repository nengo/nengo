from nengo.exceptions import TimeoutError
from nengo._vendor import portalocker


class FileLock:
    """Lock access to a file (for multithreading)."""

    def __init__(self, filename, timeout=10.0, poll=0.1):
        self.filename = filename
        self.timeout = timeout
        self.poll = poll
        self._lock = portalocker.Lock(
            self.filename, timeout=timeout, check_interval=poll, fail_when_locked=True
        )
        self._acquired = False

    def acquire(self):
        try:
            self._lock.acquire()
            self._acquired = True
        except (portalocker.AlreadyLocked, portalocker.LockException):
            raise TimeoutError(
                "Could not acquire lock '{filename}'.".format(filename=self.filename)
            )

    def release(self):
        self._lock.release()
        self._acquired = False

    @property
    def acquired(self):
        return self._acquired

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def __del__(self):
        self.release()
