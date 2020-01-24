import collections
import threading


class ThreadLocalStack(threading.local, collections.abc.Sequence):
    def __init__(self, maxsize=None):
        super().__init__()
        self.maxsize = maxsize
        self._context = []

    def __len__(self):
        return len(self._context)

    def __getitem__(self, i):
        return self._context[i]

    def append(self, item):
        if self.maxsize is not None and len(self) >= self.maxsize:
            raise RuntimeError("Stack limit exceeded.")
        self._context.append(item)

    def pop(self):
        return self._context.pop()

    def clear(self):
        self._context[:] = []
