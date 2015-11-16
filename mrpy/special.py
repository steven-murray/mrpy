"""
Definitions of all special functions used throughout ``mrpy``.
"""

import numpy as np
from mpmath import gammainc as _mp_ginc
from mpmath import gamma as _mp_g
from mpmath import hyper as _mp_hyper
from mpmath import polygamma as _mp_pg
from mpmath import meijerg as _mp_mg

# The following extends the mpmath incomplete gamma to take vector args
_ginc_vec = np.frompyfunc(lambda z, x: float(_mp_ginc(z, x)), 2, 1)
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

# The following extends the mpmath hyper function to take vector args
_hyper = np.frompyfunc(lambda a, b, z: float(_mp_hyper(a, b, z)), 3, 1)
def hyper(a,b,z):
    """
    Hypergeometric function, as defined by ``mpmath``, but modified to be able to
    take vector arguments.
    """
    hyper.__doc__ += _mp_hyper.__doc__

    if hasattr(a,"__len__") or hasattr(b,"__len__") or hasattr(z,"__len__"):
        return _hyper(a,b,z).astype("float")
    else:
        return float(_mp_hyper(a,b,z))

# The following extends the mpmath polygamma function to take vector args
_pg = np.frompyfunc(lambda a, b, z: float(_mp_pg(a, b)), 2, 1)
def polygamma(a,b):
    """
    Polygamma function, as defined by ``mpmath``, but modified to be able to
    take vector arguments.
    """
    polygamma.__doc__ += _mp_pg.__doc__

    if hasattr(a,"__len__") or hasattr(b,"__len__"):
        return _pg(a,b).astype("float")
    else:
        return float(_mp_pg(a,b))

# The following extends the mpmath MeijerG function to take vector args
_mg = np.frompyfunc(lambda a, b, z: float(_mp_mg(a, b)), 2, 1)
def meijerg(a,b,z):
    """
    Polygamma function, as defined by ``mpmath``, but modified to be able to
    take vector arguments.
    """
    meijerg.__doc__ += _mp_mg.__doc__

    if hasattr(a,"__len__") or hasattr(b,"__len__") or hasattr(z,"__len__"):
        return _mg(a,b,z).astype("float")
    else:
        return float(_mp_mg(a,b,z))

def hyperReg_vec(a, b, z):
    return hyper(a, b, z) / np.product([gammaA(bb) for bb in b])

def hyperReg(a, b, z):
    return float(hyper(a, b, z) / np.product([gamma(bb) for bb in b]))
