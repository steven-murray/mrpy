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