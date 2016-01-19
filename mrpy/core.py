"""
Basic MRP functionality, such as functions to generate the MRP with different normalisations.

Note that in this package, the `MRP` will refer to the truncated generalised gamma distribution
(TGGD), with added arbitrary normalisation. We will always directly apply it to halo mass functions,
(HMFs), and so the variate will generally be mass, *m*, and the relevant default scales will be large.

This does not in principle restrict the usage of the MRP for other applications, such as luminosity
functions or other data.
"""
import numpy as np
import special as sp
import stats
from cached_property import cached_property as _cached


def entire_integral(logHs, alpha, beta, s=1):
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

    s : array_like, optional
        Weighting (or `scaling`) of the integral.
    """
    return 10**((s + 1)*logHs)*sp.gamma((alpha + 1 + s)/beta)

def log_mass_mode(logHs,alpha,beta):
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

    >>> from mrpy import stats
    >>> from scipy.interpolate import InterpolatedUnivariateSpline as spline
    >>> m = np.linspace(13.0,14.0,200)
    >>> # Add 1 to alpha to generate mass weighting
    >>> mrp = stats.TGGDlog(14.0,-1.8+1,0.7,m[0]).pdf(m,log=True)
    >>> s = spline(m,mrp,k=4)
    >>> 10**s.derivative().roots()[0]
    1.67016715e+13
    """
    if alpha > -2:
        return 10**logHs * ((alpha+2)/beta)**(1./beta)
    elif alpha == -2:
        return np.nan
    elif alpha <-2:
        return 0

def A_rhoc(logHs, alpha, beta, Om0=0.3, rhoc=2.7755e11):
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

    Om0 : float, optional
        The matter density at redshift nought.

    rhoc : float, optional
        The critical density of the Universe.
    """
    return Om0*rhoc/entire_integral(logHs, alpha, beta, s=1)


# def log_mass_mode(logHs,alpha,beta):
#     r"""
#     The (log) mode of the MRP weighted by mass in log-space:
#     :math:`H_s \sqrt{\beta}{z+1/\beta}`.
#     """
#     return logHs + np.log10((alpha+2)/beta)/beta


def _getnorm(norm, logHs, alpha, beta, mmin, log=False, **Arhoc_kw):
    if norm == "pdf":
        x = stats.TGGD(scale=10**logHs, a=alpha, b=beta, xmin=mmin)._pdf_norm(log)
        if log:
            return -x
        else:
            return 1./x
    elif np.all(np.isreal(norm)):
        if log:
            return np.log(norm)
        else:
            return norm
    elif norm == "rhoc":
        x =  A_rhoc(logHs, alpha, beta, **Arhoc_kw)
        if log:
            return np.log(x)
        else:
            return x
    else:
        ValueError("norm should be a float, or the strings 'pdf' or 'rhoc'")


def _head(m, logHs, alpha, beta, mmin=None, norm="pdf", log=False, **Arhoc_kw):
    if mmin is None:
        mmin = m.min()

    tggd = stats.TGGD(a=alpha, b=beta, xmin=mmin, scale=10**logHs)

    A = _getnorm(norm, logHs, alpha, beta, mmin, log, **Arhoc_kw)
    return tggd, A


def _tail(shape, A, log):
    if log:
        return shape + A
    else:
        return shape*A


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

    \*\*Arhoc_kw :
        Arguments directly forwarded to the mean-density normalisation, :func:`A_rhoc`.
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


def ngtm(m, logHs, alpha, beta, mmin=None, mmax=np.inf, norm="pdf", log=False, **Arhoc_kw):
    """
    The integral of the MRP, in reverse (i.e. CDF=1 at mmin).

    %s
    """
    t, A = _head(m, logHs, alpha, beta, mmin, norm, log, **Arhoc_kw)
    t = stats.TGGD(a=alpha, b=beta, xmin=m, scale=10**logHs)
    shape = t._pdf_norm(log)

    return _tail(shape, A, log)


ngtm.__doc__ %= _pardoc


def rho_gtm(m, logHs, alpha, beta, mmin=None, mmax=np.inf, norm="pdf", log=False, **Arhoc_kw):
    """
    The mass-weighted integral of the MRP, in reverse (ie. from high to low mass)

    %s
    """
    t, A = _head(m, logHs, alpha, beta, mmin, norm, log, **Arhoc_kw)
    shape = 10**(2*logHs)*sp.gammainc((alpha + 2)/beta, (m/10**logHs)**beta)
    if log:
        shape = np.log(shape)
    return _tail(shape, A, log)


rho_gtm.__doc__ %= _pardoc


# def get_alpha_and_A(logHs, beta, mmin, mw_integ, Om0, s=0, rhoc=2.7755e11):
#     r"""
#     Recover alpha and normalisation, A, given known total mass density, and
#     given (mass-scaled) integral of data down to mmin. Specifically, solve the system:
#
#     ..math :: A = \frac{\Omega_m \rho_c}{\H_s^2 \Gamma(\frac{\alpha+2}{\beta})}
#
#     ..math :: A = \frac{I_s}{H_s^{s+1}\Gamma(\frac{\alpha+s+1}{\beta},(m/H_s)^\beta)},
#
#     where :math:`I_s` is the integral of the mass function (specified by the data),
#     multiplied by a mass-weight, i.e., :math:`m^s`.
#
#     Combining the two equations, we can solve for :math:`\alpha`, but the solution
#     is gained by root-finding using Newton's method.
#
#     ..note :: This routine necessarily returns an alpha greater than -2, otherwise
#               the total mass density is undefined.
#     """
#     rhomean = rhoc*Om0
#
#     def f(lnz):
#         """
#         Function whose root will yield correct alpha.
#
#         Input is lnz = ln((alpha+2)/beta)
#         """
#         return rhomean/gamma(np.exp(lnz)) - mw_integ/(
#         10**(logHs*(s - 1))*gammainc(np.exp(lnz) + (s - 1)/beta, 10**(beta*(mmin - logHs))))
#
#     lnz = newton(f, np.log(0.1/beta))
#     alpha = np.exp(lnz)*beta - 2
#     A = A_rhoc(logHs, alpha, beta, Om0, rhoc)
#
#     return alpha, A




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
        render the MRP as a statistical distribution. If set to ``"rhoc"``, it will
        yield the correct total mass density across all masses, down to ``m=0``.

    log_mmin : array_like, optional
        Log-10 truncation mass of the MRP. By default is set to the minimum mass
        in ``logm``.

    Om0 : float, optional
        Matter density of the Universe. Only required if `norm` is set to ``Arhoc``.

    rhoc : float, optional
        Crtical density of the Universe. Only required if `norm` is set to ``Arhoc``.
    """

    def __init__(self, logm, logHs, alpha, beta, norm="pdf",log_mmin=None,
                 Om0=0.3, rhoc=2.7755e11):
        self.logm = logm
        if log_mmin is not None:
            self.log_mmin = log_mmin
        else:
            try:
                self.log_mmin = logm.min()
            except:
                self.log_mmin = logm

        self.logHs = logHs
        self.alpha = alpha
        self.beta = beta
        self._norm = norm
        self._Arhoc_kw = {"Om0":Om0,"rhoc":rhoc}

    @property
    def m(self):
        """
        Real-space masses
        """
        return 10**self.logm

    @property
    def mmin(self):
        """
        Real-space truncation mass
        """
        return 10**self.log_mmin

    @property
    def Hs(self):
        """
        Real-space scale mass.
        """
        return 10**self.logHs

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
                        self.mmin, log=True, **self._Arhoc_kw)

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

    def rho_gtm(self,log=False):
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
    def log_mass_mode(self):
        """
        The mode of the log-space MRP weighted by mass
        """
        return log_mass_mode(self.logHs, self.alpha, self.beta)

    @_cached
    def _k(self):
        """
        The integral of the mass-weighted MRP down to ``m=0`` (i.e. disregarding
        the truncation mass).
        """
        return entire_integral(self.logHs, self.alpha, self.beta)


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
