"""
Definitions of all special functions used throughout ``mrpy``.
"""

import numpy as np
from mpmath import gammainc as _mp_ginc
from mpmath import gamma as _mp_g

# The following extends the mpmath incomplete gamma to take vector args
_ginc_vec = np.frompyfunc(lambda z, x: float(mp_ginc(z, x)), 2, 1)
def gammainc(z, x):
    """
    Incomplete gamma function, as defined by ``mpmath``, but modified to be able
    to take vector arguments.

    """
    gammainc.__doc__ += _mp_ginc.__doc__

    if hasattr(z, "__len__") or hasattr(x, "__len__"):
        return _ginc_vec(z, x).astype("float")
    else:
        return float(_mp_ginc(z, x))


# The following extends the mpmath gamma to take vector args
_g_vec = np.frompyfunc(lambda z: float(_mp_g(z)), 1, 1)
def gamma(z):
    """
    Gamma function, as defined by ``mpmath``, but modified to be able
    to take vector arguments.
    """
    gamma.__doc__ += _mp_g.__doc__

    if hasattr(z, "__len__"):
        return _g_vec(z).astype("float")
    else:
        return float(_mp_g(z))
