import os
from . import exceptions
from . import constants


if os.name == 'nt':  # pragma: no cover
    import msvcrt

    def lock(file_, flags):
        if flags & constants.LOCK_SH:
            if flags & constants.LOCK_NB:
                mode = msvcrt.LK_NBRLCK
            else:
                mode = msvcrt.LK_RLOCK
        else:
            if flags & constants.LOCK_NB:
                mode = msvcrt.LK_NBLCK
            else:
                mode = msvcrt.LK_LOCK

        # windows locks byte ranges, so make sure to lock from file start
        try:
            savepos = file_.tell()
            file_.seek(0, os.SEEK_END)
            if file_.tell() > 0x7fffffff:
                raise exceptions.FileToLarge("Files larger than 2GB "
                                             "not supported on Windows")
            # [ ] test exclusive lock fails on seek here
            # [ ] test if shared lock passes this point
            file_.seek(0)
            try:
                # [x] check if 0 param locks entire file (not documented in
                #     Python)
                # [x] just fails with "IOError: [Errno 13] Permission denied",
                # [x] -1 crashes on py3.5+
                # [x] larger than 0x7fffffff gives OverflowError
                # [ ] writing past 2GB goes to unlocked space, should not
                #     be a problem when only using portalocker, since it
                #     always locks from the start
                msvcrt.locking(file_.fileno(), mode, 0x7fffffff)
            except IOError as exc_value:
                # [ ] be more specific here
                raise exceptions.LockException(
                    exceptions.LockException.LOCK_FAILED, exc_value.strerror)
            finally:
                if savepos:
                    file_.seek(savepos)
        except IOError as exc_value:
            raise exceptions.LockException(
                exceptions.LockException.LOCK_FAILED, exc_value.strerror)

    def unlock(file_):
        try:
            savepos = file_.tell()
            if savepos:
                file_.seek(0)
            try:
                msvcrt.locking(file_.fileno(), constants.LOCK_UN, 0x7fffffff)
            except IOError as exc_value:
                raise exceptions.LockException(
                    exceptions.LockException.LOCK_FAILED, exc_value.strerror)
            finally:
                if savepos:
                    file_.seek(savepos)
        except IOError as exc_value:
            raise exceptions.LockException(
                exceptions.LockException.LOCK_FAILED, exc_value.strerror)

elif os.name == 'posix':  # pragma: no cover
    import fcntl

    def lock(file_, flags):
        try:
            fcntl.flock(file_.fileno(), flags)
        except IOError as exc_value:
            # The exception code varies on different systems so we'll catch
            # every IO error
            raise exceptions.LockException(exc_value)

    def unlock(file_):
        fcntl.flock(file_.fileno(), constants.LOCK_UN)

else:  # pragma: no cover
    raise RuntimeError('PortaLocker only defined for nt and posix platforms')

