"""
Definitions of all special functions used throughout `mrpy`.

Generally, these are adapted from `mpmath`, but return standard floats/arrays, and can take
`array_like` input.
"""

import numpy as np
from mpmath import gammainc as _mp_ginc
from mpmath import gamma as _mp_g
from mpmath import hyper as _mp_hyper
from mpmath import polygamma as _mp_pg
from mpmath import meijerg as _mp_mg

docs = """
    {0} function.

    .. note:: This is exactly as defined by `mpmath`, but modified to take
              ``array_like`` arguments, and return results with type `float`.

    Notes
    -----
    {1}
"""
def _flt(a):
    try:
        return a.astype('float')
    except AttributeError:
        return float(a)

# The following extends the mpmath incomplete gamma to take vector args
_ginc_ufunc = np.frompyfunc(lambda z, x: _mp_ginc(z, x), 2, 1)
def gammainc(z,x):
    return _flt(_ginc_ufunc(z,x))
gammainc.__doc__ =  docs.format("Upper incomplete gamma",_mp_ginc.__doc__)


# The following extends the mpmath gamma to take vector args
_g_ufunc = np.frompyfunc(lambda z: _mp_g(z), 1, 1)
def gamma(z):
    return _flt(_g_ufunc(z))
gamma.__doc__ =  docs.format("Gamma",_mp_g.__doc__)

# The following extends the mpmath polygamma function to take vector args
_pg_ufunc = np.frompyfunc(lambda a, b: _mp_pg(a, b), 2, 1)
def polygamma(m,z):
    return _flt(_pg_ufunc(m,z))
polygamma.__doc__ = docs.format("Polygamma",_mp_pg.__doc__)


# The following extends the mpmath meijerg function to take vector args
_g1_ufunc = np.frompyfunc(lambda z,x: _mp_mg([[], [1, 1]], [[0, 0, z], []], x)/gammainc(z,x),2,1)
def G1(z,x):
    r"""
    The Meijer-G function with specific arguments: ``meijerg([[], [1, 1]], [[0, 0, z], []], x)``,
    normalised by the incomplete gamma function with arguments `z,x`.

    Either `z` or `x` can be `array_like`.

    Notes
    -----
    This quantity arises in the derivative of the natural log of the incomplete gamma function.

    .. math:: \frac{d}{dz} \ln \Gamma(z,x) = G1(z,x) + \ln x
    """
    return _flt(_g1_ufunc(z,x))

_g2_ufunc = np.frompyfunc(lambda z,x: _mp_mg([[], [1, 1,1]], [[0, 0,0, z], []], x)/gammainc(z,x),2,1)
def G2(z,x):
    r"""
    The Meijer-G function with specific arguments: ``meijerg([[], [1, 1,1]], [[0, 0, 0, z], []], x)``,
    normalised by the incomplete gamma function with arguments `z,x`.

    Either `z` or `x` can be `array_like`.

    Notes
    -----
    This quantity arises in the derivative of :func:`G1`.
    """
    return _flt(_g2_ufunc(z,x))

# The following extends the mpmath hyper function to take vector args
_hyperReg_2F2_ufunc = np.frompyfunc(lambda z,x: _mp_hyper([z,z], [z+1,z+1], -x)/gamma(z+1)**2,2,1)
def hyperReg_2F2(z,x):
    """
    The regularised hypergeometric function with specific arguments: ``hyper([z,z],[z+1,z+1],-x)``.

    Parameters
    ----------
    z : array_like
    x : array_like

    Returns
    -------
    array_like
    """
    return _flt(_hyperReg_2F2_ufunc(z,x))