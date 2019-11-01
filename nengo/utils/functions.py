def function_name(func):
    """Returns the name of a function.

    Unlike accessing ``func.__name__``, this function is robust to the
    different types of objects that can be considered a function in Nengo.

    Parameters
    ----------
    func : callable or array_like
        Object used as function argument.

    Returns
    -------
    str
        Name of function object.
    """
    return getattr(func, "__name__", func.__class__.__name__)
