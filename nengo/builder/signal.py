"""Signals represent values that will be used in the simulation.

This code adapted from sigops/signal.py and sigops/signaldict.py
(https://github.com/jaberg/sigops).
This modified code is included under the terms of their license:

Copyright (c) 2014, James Bergstra
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np


class Signal(object):
    """Interpretable, vector-valued quantity within Nengo"""

    # Set assert_named_signals True to raise an Exception
    # if model.signal is used to create a signal with no name.
    # This can help to identify code that's creating un-named signals,
    # if you are trying to track down mystery signals that are showing
    # up in a model.
    assert_named_signals = False

    def __init__(self, value, name=None, base=None):
        # Make sure we use a C-contiguous array
        self._value = np.array(value, copy=(base is None), order='C',
                               dtype=np.float64)
        self._value.flags.writeable = np.asarray(value).flags.writeable

        if self._value.flags.writeable:
            self.init_val = np.array(value, copy=True, order='C',
                                     dtype=np.float64)
            self.init_val.flags.writeable = False

        self._base = base
        if name is not None:
            self._name = name
        if Signal.assert_named_signals:
            assert name

    def __str__(self):
        try:
            return "Signal(" + self._name + ", shape=" + str(self.shape) + ")"
        except AttributeError:
            return ("Signal(id " + str(id(self)) + ", shape="
                    + str(self.shape) + ")")

    def __repr__(self):
        return str(self)

    def reset(self):
        if self._value.flags.writeable:
            self.value = self.init_val

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    @property
    def size(self):
        return self.value.size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            return '<Signal%d>' % id(self)

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def base(self):
        return self if self._base is None else self._base

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value[...] = val

    def __getitem__(self, item):
        # indexing/slicing into array
        return Signal(self._value[item],
                      name="%s[%s]" % (self.name, item),
                      base=self)

    def reshape(self, *shape):
        return Signal(self._value.reshape(*shape),
                      name="%s.reshape(%s)" % (self.name, shape),
                      base=self)
