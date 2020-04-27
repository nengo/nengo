"""Functions for filter design.

These functions are now located in `_vendor/scipy/signal` to reflect the fact that they
are copied from Scipy. They are imported here for backwards compatibility.
"""

from nengo._vendor.scipy.signal import (  # pylint: disable=unused-import
    abcd_normalize,
    cont2discrete,
    normalize,
    ss2tf,
    ss2zpk,
    tf2ss,
    tf2zpk,
    zpk2ss,
    zpk2tf,
)
