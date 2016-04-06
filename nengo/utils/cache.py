"""Utilities to convert to and from bytes.

Used by nengo.runcom in order to present file sizes to users in
human-readable formats.

This code adapted from http://goo.gl/zeJZl under the MIT License.
"""


def bytes2human(n, fmt="%(value).1f %(symbol)s"):
    """Convert from a size in bytes to a human readable string.

    Examples
    --------
    >>> bytes2human(10000)
    '9 KB'
    >>> bytes2human(100001221)
    '95 MB'
    """
    symbols = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return fmt % {'value': value, 'symbol': symbol}
    return fmt % {'value': n, 'symbol': symbols[0]}


def human2bytes(s):
    """Convert from a human readable string to a size in bytes.

    Examples
    --------
    >>> human2bytes('1 MB')
    1048576
    >>> human2bytes('1 GB')
    1073741824
    """
    symbols = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    letter = s[-2:].strip().upper()
    num = s[:-2].strip()
    assert letter in symbols
    num = float(num)
    prefix = {symbols[0]: 1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    return int(num * prefix[letter])


def byte_align(size, alignment):
    """Returns the int larger than ``size`` aligned to ``alginment`` bytes."""
    mask = alignment - 1
    if size & mask == 0:
        return size
    else:
        return (size | mask) + 1
