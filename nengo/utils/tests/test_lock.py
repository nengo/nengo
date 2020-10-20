import multiprocessing
import sys
import time

import pytest

from nengo.utils.lock import FileLock, TimeoutError


def acquire_lock(filename):
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    try:
        lock.acquire()
    except TimeoutError:
        sys.exit(0)
    else:
        sys.exit(1)


def acquire_lock_and_idle(filename):
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    while True:
        time.sleep(1.0)


def test_can_acquire_filelock_at_most_once(tmp_path):
    filename = tmp_path / "lock"
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    p = multiprocessing.Process(target=acquire_lock, args=(filename,))
    p.start()
    p.join()
    lock.release()
    assert p.exitcode == 0


@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_process_termination_releases_lock(tmp_path):
    filename = tmp_path / "lock"
    p = multiprocessing.Process(target=acquire_lock_and_idle, args=(filename,))
    p.start()
    while p.is_alive() and not filename.exists():
        time.sleep(0.2)
    assert p.is_alive()

    lock = FileLock(filename, timeout=0.01, poll=0.01)
    with pytest.raises(TimeoutError):
        lock.acquire()
    p.terminate()
    p.join()
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    lock.release()


def test_released_filelock_can_be_reacquired(tmp_path):
    filename = tmp_path / "lock"
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    lock.release()
    lock.acquire()
    lock.release()


def test_can_release_filelock_multiple_times(tmp_path):
    filename = tmp_path / "lock"
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.release()
    lock.acquire()
    lock.release()
    lock.release()


def test_filelock_supports_with_statement(tmp_path):
    filename = tmp_path / "lock"
    with FileLock(filename):
        pass


def test_filelock_gets_released_on_lock_deletion(tmp_path):
    filename = tmp_path / "lock"
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    del lock
    lock = FileLock(filename, timeout=0.01, poll=0.01)
    lock.acquire()
    lock.release()
