from nengo.exceptions import MovedError


def spikes2events(*args, **kwargs):
    """Moved to nengo_extras.neurons."""
    raise MovedError(location="nengo_extras.neurons")


def _rates_isi_events(*args, **kwargs):
    """Moved to nengo_extras.neurons."""
    raise MovedError(location="nengo_extras.neurons")


def rates_isi(*args, **kwargs):
    """Moved to nengo_extras.neurons."""
    raise MovedError(location="nengo_extras.neurons")


def lowpass_filter(*args, **kwargs):
    """Moved to nengo_extras.neurons."""
    raise MovedError(location="nengo_extras.neurons")


def rates_kernel(*args, **kwargs):
    """Moved to nengo_extras.neurons."""
    raise MovedError(location="nengo_extras.neurons")


def settled_firingrate(*args, **kwargs):
    """Moved to nengo.neurons."""
    raise MovedError(location="nengo.neurons")
