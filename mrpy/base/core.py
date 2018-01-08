"""
Basic MRP functionality, such as functions to generate the MRP with different normalisations.

Note that in this package, the `MRP` will refer to the truncated generalised gamma distribution
(TGGD), with added arbitrary normalisation. We will always directly apply it to halo mass functions,
(HMFs), and so the variate will generally be mass, *m*, and the relevant default scales will be large.

This does not in principle restrict the usage of the MRP for other applications, such as luminosity
functions or other data.
"""
import numpy as np
import mrpy.base.special as sp
from . import stats


def entire_integral(logHs, alpha, beta):
    r"""
    The entire integral of the un-normalised mass-weighted *non-truncated* MRP:

    .. math:: \int_0^\infty m^s f(m) = \mathcal{H}_\star^{s+1} \Gamma\left(\frac{\alpha+1+s}{\beta}\right) \ dm.

    where *s* defines a weighting of the integral, in which the immediate application is that
    ``s=1`` gives the total mass density.

    .. note:: The sum of `alpha` and `s` must be greater than -1.

    Parameters
    ----------
    logHs : array_like
        The base-10 logarithm of the scale mass, :math:`H_\star`.

    alpha : array_like
        The power-law index

    beta : array_like
        Exponential cutoff parameter
    """
    return 10 ** (2 * logHs) * sp.gamma((alpha + 2) / beta)


def log_mass_mode(logHs, alpha, beta):
    """
    The mode of the log-space MRP weighted by mass.

    Parameters
    ----------
    logHs, alpha, beta: array_like
        Shape parameters of the MRP distribution.

    Returns
    -------
    lmm : array_like
        The log-space mass mode of the MRP.

    Examples
    --------
    This function:

    >>> log_mass_mode(14.0,-1.8,0.7)
    1.67016714698e+13

    yields the same result as generating the mode via differentiation:

    >>> from mrpy.base import stats
    >>> from scipy.interpolate import InterpolatedUnivariateSpline as spline
    >>> m = np.linspace(13.0,14.0,200)
    >>> # Add 1 to alpha to generate mass weighting
    >>> mrp = stats.TGGDlog(14.0,-1.8+1,0.7,m[0]).pdf(m,log=True)
    >>> s = spline(m,mrp,k=4)
    >>> 10**s.derivative().roots()[0]
    1.67016715e+13
    """
    if alpha > -2:
        return 10 ** logHs * ((alpha + 2) / beta) ** (1. / beta)
    elif alpha == -2:
        return np.nan
    elif alpha < -2:
        return 0


def A_rhom(logHs, alpha, beta, rhom=0.3 * 2.7755e11):
    r"""
    The normalisation required to bind all matter in halos of some scale:

    .. math:: A_{\rho_c} = \Omega_m \rho_c / k(\vec{\theta})

    where *k* is the :func:`entire_integral` of the MRP, with a mass-weighting
    (or scaling) of ``s=1``.

    Parameters
    ----------
    logHs : array_like
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : array_like
        The power-law index

    beta : array_like
        Exponential cutoff parameter

    rhom : float, optional
        The mass density of the Universe.
    """
    return rhom / entire_integral(logHs, alpha, beta)


# def log_mass_mode(logHs,alpha,beta):
#     r"""
#     The (log) mode of the MRP weighted by mass in log-space:
#     :math:`H_s \sqrt{\beta}{z+1/\beta}`.
#     """
#     return logHs + np.log10((alpha+2)/beta)/beta


def _getnorm(norm, logHs, alpha, beta, mmin, log=False, **Arhom_kw):
    if norm == "pdf":
        x = stats.TGGD(scale=10 ** logHs, a=alpha, b=beta, xmin=mmin)._pdf_norm(log)
        if log:
            return -x
        else:
            return 1. / x
    elif np.all(np.isreal(norm)):
        if log:
            return np.log(norm)
        else:
            return norm
    elif norm == "rhom":
        x = A_rhom(logHs, alpha, beta, **Arhom_kw)
        if log:
            return np.log(x)
        else:
            return x
    else:
        ValueError("norm should be a float, or the strings 'pdf' or 'rhom'")


def _head(m, logHs, alpha, beta, mmin=None, norm="pdf", log=False, **Arhoc_kw):
    if mmin is None:
        mmin = m.min()

    tggd = stats.TGGD(a=alpha, b=beta, xmin=mmin, scale=10 ** logHs)

    A = _getnorm(norm, logHs, alpha, beta, mmin, log, **Arhoc_kw)
    return tggd, A


def _tail(shape, A, log):
    if log:
        return shape + A
    else:
        return shape * A


_pardoc = """
    Parameters
    ----------
    m : array_like
        Vector of masses at which to evaluate the MRP

    logHs : float
        The base-10 logarithm of the scale mass, :math:`H_\star`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter

    mmin : float, optional
        The lower-truncation mass. Default is the minimum mass in ``m``.

    norm :string or float, optional
        Gives the normalisation of the MRP, *A*. If set to a *float*, it
        is directly the normalisation. If set to ``"pdf"``, it will automatically
        render the MRP as a statistical distribution. If set to ``"rhoc"``, it will
        yield the correct total mass density across all masses, down to ``m=0``.

    log : logical
        Whether to return the natural log of the MRP (suitable for Bayesian
        likelihoods).

    \*\*Arhom_kw :
        Arguments directly forwarded to the mean-density normalisation, :func:`A_rhom`.
    """


def dndm(m, logHs, alpha, beta, mmin=None, norm="pdf", log=False, **Arhoc_kw):
    """
    The MRP distribution.

    %s
    """
    t, A = _head(m, logHs, alpha, beta, mmin, norm, log, **Arhoc_kw)
    shape = t._pdf_shape(m, log)
    return _tail(shape, A, log)


dndm.__doc__ %= _pardoc


def ngtm(m, logHs, alpha, beta, mmin=None, norm="pdf", log=False, **Arhom_kw):
    """
    The integral of the MRP, in reverse (i.e. CDF=1 at mmin).

    %s
    """
    t, A = _head(m, logHs, alpha, beta, mmin, norm, log, **Arhom_kw)
    t = stats.TGGD(a=alpha, b=beta, xmin=m, scale=10 ** logHs)
    shape = t._pdf_norm(log)

    return _tail(shape, A, log)


ngtm.__doc__ %= _pardoc


def rho_gtm(m, logHs, alpha, beta, mmin=None, norm="pdf", log=False, **Arhom_kw):
    """
    The mass-weighted integral of the MRP, in reverse (ie. from high to low mass)

    %s
    """
    _, A = _head(m, logHs, alpha, beta, mmin, norm, log, **Arhom_kw)
    shape = 10 ** (2 * logHs) * sp.gammainc((alpha + 2) / beta, (m / 10 ** logHs) ** beta)
    if log:
        shape = np.log(shape)
    return _tail(shape, A, log)


rho_gtm.__doc__ %= _pardoc


class MRP(object):
    """
    An MRP object.

    This class contains methods for calculating typical quantities of interest:
    the differential/cumulative number densities, as well as mass densities, and
    several types of normalisation. Also included is a pointer to underlying
    statistical quantities, such as mean, median, mode etc.

    Parameters
    ----------
    logm : array_like
        Vector of log10 masses.

    logHs, alpha, beta : array_like
        The shape parameters of the MRP.

    norm : float or string

        Gives the normalisation of the MRP, *lnA*. If set to a *float*, it
        is directly the (log) normalisation. If set to ``"pdf"``, it will automatically
        render the MRP as a statistical distribution. If set to ``"rhom"``, it will
        yield the correct total mass density across all masses, down to ``M=0``.

    log_mmin : array_like, optional
        Log-10 truncation mass of the MRP. By default is set to the minimum mass
        in ``logm``.

    rhom : float, optional
        Mass density of the Universe. Only required if `norm` is set to ``Arhom``.
    """

    def __init__(self, logm, logHs, alpha, beta, norm="pdf", log_mmin=None,
                 rhom=0.3 * 2.7755e11):
        self.logm = logm
        if log_mmin is not None:
            self.log_mmin = log_mmin
        else:
            try:
                self.log_mmin = logm.min()
            except AttributeError:
                self.log_mmin = logm

        self.logHs = logHs
        self.alpha = alpha
        self.beta = beta
        self._norm = norm
        self._Arhom_kw = {"rhom": rhom}

    @property
    def m(self):
        """
        Real-space masses
        """
        return 10 ** self.logm

    @property
    def mmin(self):
        """
        Real-space truncation mass
        """
        return 10 ** self.log_mmin

    @property
    def Hs(self):
        """
        Real-space scale mass.
        """
        return 10 ** self.logHs

    @property
    def stats(self):
        """
        An object containing statistical quantities of the MRP.

        This is basically a :class:`mrpy.stats.TGGD` class.
        """
        return stats.TGGD(scale=self.Hs, a=self.alpha, b=self.beta, xmin=self.mmin)

    @property
    def lnA(self):
        """
        Natural log of the normalisation
        """
        if np.lib._iotools._is_string_like(self._norm):
            norm = self._norm
        else:
            norm = np.exp(self._norm)

        return _getnorm(norm, self.logHs, self.alpha, self.beta,
                        self.mmin, log=True, **self._Arhom_kw)

    @property
    def A(self):
        """Normalisation of the MRP"""
        return np.exp(self.lnA)

    # =============================================================================
    # Principal Vector Quantities
    # =============================================================================
    def dndm(self, log=False):
        """
        Return the MRP at `m`.

        Parameters
        ----------
        log : logical, optional
            Whether to return the natural log of the MRP.
        """
        return dndm(self.m, self.logHs, self.alpha, self.beta, mmin=self.log_mmin,
                    norm=self.A, log=log)

    def dndlog10m(self, log=False):
        """
        Return the MRP in log10 space at `m'.

        Parameters
        ----------
        log : logical, optional
            Whether to return the natural log of the MRP.
        """
        if not log:
            return self.dndm(log) * self.m * np.log(10)
        else:
            return self.dndm(log) + np.log(10) * self.logm + np.log(np.log(10))

    def ngtm(self, log=False):
        """
        The number density greater than `mmin`.

        Parameters
        ----------
        log : logical
            Whether to return the natural log of the number density.
        """
        return ngtm(self.m, self.logHs, self.alpha, self.beta, mmin=self.log_mmin,
                    norm=self.A, log=log)

    def rho_gtm(self, log=False):
        """
        The mass-weighted integral of the MRP, in reverse (ie. from high to low mass).


        Parameters
        ----------
        log : logical
            Whether to return the natural log of the density.
        """
        return rho_gtm(self.m, self.logHs, self.alpha, self.beta, mmin=self.log_mmin,
                       norm=self.A, log=log)

    # =============================================================================
    # Derived Scalar Quantities
    # =============================================================================
    @property
    def _k(self):
        """
        The integral of the mass-weighted MRP down to ``M=0`` (i.e. disregarding
        the truncation mass).
        """
        return entire_integral(self.logHs, self.alpha, self.beta)

    @property
    def log_mass_mode(self):
        """
        The mode of the log-space MRP weighted by mass
        """
        return log_mass_mode(self.logHs, self.alpha, self.beta)

    @property
    def nbar(self):
        """
        Total number density above truncation mass.
        """
        return ngtm(self.mmin, self.logHs, self.alpha, self.beta, mmin=self.log_mmin,
                    norm=self.A, log=False)

    @property
    def rhobar(self):
        """
        Total mass density above truncation mass.
        """
        return rho_gtm(self.mmin, self.logHs, self.alpha, self.beta, mmin=self.log_mmin,
                       norm=self.A, log=False)
