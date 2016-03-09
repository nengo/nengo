"""Implementation of the Nengo cache object (NCO) protocol.

Nengo cache objects store a Numpy array and some associated, picklable Python
object in a single, uncompressed file. These files are not platform independent
as they are optimized for fast reading and writing, and cached data is not
supposed to be shared across platforms.

The protocol version 0 is as follows:

* A header consisting of:
    * 3 bytes with the magic string 'NCO'
    * 1 unsigned byte indicating the protocol version
    * unsigned long int denoting the start of the Python object data
    * unsigned long int denoting the end of the Python object data
    * unsigned long int denoting the start of the array data
    * unsigned long int denoting the end of the array data
* Potentially some padding bytes.
* The Python object data pickled by the (c)pickle module using the highest
  available protocol.
* Potentially some padding bytes.
* The array data in NPY format.

Files will be written with padding to have both the Python object data and the
array data an alignment of 16 bytes.

The Numpy NPY format is documented here:
https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst
"""

from __future__ import absolute_import

import os
import struct

import numpy as np

from .compat import ensure_bytes, pickle
from .cache import byte_align
from ..exceptions import CacheIOError


class Subfile(object):
    """A file-like object for limiting reads to a subrange of a file.

    This clss only supports reading and seeking. Writing is not supported.

    Parameters
    ----------
    fileobj : file-like object
        Complete files.
    start : int
        Offset of the first readable position in the file.
    end : int
        Offset of the last readable position + 1 in the file.
    """
    def __init__(self, fileobj, start, end):
        self.fileobj = fileobj
        self.start = start
        self.end = end
        self.max_size = end - start

        self.fileobj.seek(start)

    def read(self, size=None):
        size = min(size, self.max_size) if size is not None else self.max_size
        self.max_size -= size
        return self.fileobj.read(size)

    def readline(self, size=None):
        size = min(size, self.max_size) if size is not None else self.max_size
        data = self.fileobj.readline(size)
        self.max_size = self.end - self.fileobj.tell()
        return data

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset = self.fileobj.tell() + offset
        elif whence == os.SEEK_SET:
            offset = self.start + offset
        elif whence == os.SEEK_END:
            offset = self.end + offset
        else:
            raise NotImplementedError()
        offset = max(self.start, min(self.end, offset))
        self.max_size = self.end - offset
        self.fileobj.seek(offset)


MAGIC_STRING = ensure_bytes('NCO')
SUPPORTED_PROTOCOLS = [0]
HEADER_FORMAT = '@{}sBLLLL'.format(len(MAGIC_STRING))
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
ALIGNMENT = 16


def write(fileobj, metadata, array):
    """Writes a Nengo cache object.

    Parameters
    ----------
    fileobj : file-like object
        File object to write the data to.
    metadata : object
        Python object with metadata (will be pickled).
    array : ndarray
        Numpy array with the actual data to store.
    """
    pickle_start = byte_align(HEADER_SIZE, ALIGNMENT)
    fileobj.seek(pickle_start)
    pickle.dump(metadata, fileobj, pickle.HIGHEST_PROTOCOL)
    pickle_end = fileobj.tell()

    array_start = byte_align(pickle_end, ALIGNMENT)
    fileobj.seek(array_start)
    np.save(fileobj, array)
    array_end = fileobj.tell()

    header = struct.pack(
        HEADER_FORMAT, MAGIC_STRING, 0, pickle_start, pickle_end,
        array_start, array_end)
    fileobj.seek(0)
    fileobj.write(header)


def read(fileobj):
    """Reads a Nengo cache object.

    Parameters
    ----------
    fileobj : file-like object
        The file object to read from.

    Returns
    -------
    metadata, array
        Returns a tuple with the Python object containing the metadata as first
        element and the array with the actual data as second element.
    """
    header = fileobj.read(HEADER_SIZE)
    magic, version, pickle_start, pickle_end, array_start, array_end = (
        struct.unpack(HEADER_FORMAT, header))

    if magic != MAGIC_STRING:
        raise CacheIOError("Not a Nengo cache object file.")
    if version not in SUPPORTED_PROTOCOLS:
        raise CacheIOError(
            "NCO protocol version {} is not supported.".format(version))

    metadata = pickle.load(Subfile(fileobj, pickle_start, pickle_end))
    array = np.load(Subfile(fileobj, array_start, array_end))
    return metadata, array
