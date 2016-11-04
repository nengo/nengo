from nengo.exceptions import MovedError


def probe_all(*args, **kwargs):
    raise MovedError(location="nengo_extras.probe")
