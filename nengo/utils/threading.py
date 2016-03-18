from __future__ import absolute_import

import collections
import threading


class ThreadLocalStack(threading.local, collections.Sequence):
    def __init__(self):
        super(ThreadLocalStack, self).__init__()
        self._context = []

    def __len__(self):
        return len(self._context)

    def __getitem__(self, i):
        return self._context[i]

    def append(self, item):
        self._context.append(item)

    def pop(self):
        return self._context.pop()

    def clear(self):
        self._context[:] = []
