import os.path

import pytest

from nengo.utils.lock import FileLock, TimeoutError


def test_can_acquire_filelock_at_most_once(tmpdir):
    filename = os.path.join(str(tmpdir), 'lock')
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    with pytest.raises(TimeoutError):
        lock.acquire()
    lock2 = FileLock(filename, timeout=0.01, poll=0.01)
    with pytest.raises(TimeoutError):
        lock2.acquire()
    lock.release()


def test_released_filelock_can_be_reacquired(tmpdir):
    filename = os.path.join(str(tmpdir), 'lock')
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    lock.release()
    lock.acquire()
    lock.release()


def test_can_release_filelock_multiple_times(tmpdir):
    filename = os.path.join(str(tmpdir), 'lock')
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.release()
    lock.acquire()
    lock.release()
    lock.release()


def test_filelock_supports_with_statement(tmpdir):
    filename = os.path.join(str(tmpdir), 'lock')
    with FileLock(filename):
        pass


def test_filelock_gets_released_on_lock_deletion(tmpdir):
    filename = os.path.join(str(tmpdir), 'lock')
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    del lock
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    lock.release()
