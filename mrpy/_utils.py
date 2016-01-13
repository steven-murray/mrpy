import numpy as np

def copydoc(fromfunc, sep="\n"):
    """
    Decorator: Copy the docstring of `fromfunc`

    Shamelessly adapted from
    http://stackoverflow.com/questions/13741998/is-there-a-way-to-let-classes-inherit-the-documentation-of-their-superclass-with?lq=1
    """
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ == None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func
    return _decorator


def insertdoc(str, sep="\n"):
    """
    Decorator: Insert the string `str` into the doc of the decorated function
    at positions `%s`.

    Shamelessly adapted from
    http://stackoverflow.com/questions/13741998/is-there-a-way-to-let-classes-inherit-the-documentation-of-their-superclass-with?lq=1
    """
    def _decorator(func):
        if func.__doc__ == None:
            func.__doc__ = str
        else:
            func.__doc__ %= str
        return func
    return _decorator




def numerical_jac(func, keys, dx=1e-4, **kwargs):
    y0 = func(**kwargs)
    out = np.zeros(len(keys))
    for i, k in enumerate(keys):
        kwargs[k] += dx
        out[i] = func(**kwargs) - y0
        kwargs[k] -= dx
    return out/dx


def numerical_hess(func, keys, dx=1e-5, **kwargs):
    j0 = numerical_jac(func, keys, dx, **kwargs)
    out = np.zeros((len(keys), len(keys)))
    for i, k in enumerate(keys):
        kwargs[k] += dx
        out[i, :] = numerical_jac(func, keys, dx, **kwargs) - j0
        kwargs[k] -= dx
    return out/dx

def extra_squeeze(a):
    b = np.squeeze(a)
    if np.prod(b.shape)==1:
        return b[0]
    else:
        return b