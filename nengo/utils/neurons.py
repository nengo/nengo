from nengo.exceptions import MovedError


def spikes2events(*args, **kwargs):
    raise MovedError(location="nengo_extras.neurons")


def _rates_isi_events(*args, **kwargs):
    raise MovedError(location="nengo_extras.neurons")


def rates_isi(*args, **kwargs):
    raise MovedError(location="nengo_extras.neurons")


def lowpass_filter(*args, **kwargs):
    raise MovedError(location="nengo_extras.neurons")


def rates_kernel(*args, **kwargs):
    raise MovedError(location="nengo_extras.neurons")


def settled_firingrate(*args, **kwargs):
    raise MovedError(location="nengo.neurons")
