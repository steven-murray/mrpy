"""
A module defining the TGGD distribution (as well as in log and ln space) in standard R style.

See the R package `tggd` for a similar implementation.

The distribution functions implemented here are described in detail in Murray, Robotham and Power, 2016.
"""
import numpy as np
from special import gammainc
from scipy.interpolate import InterpolatedUnivariateSpline as spline

ln10 = np.log(10)


# ------------- PDFs --------------------------------------------
def _dtggd(x, scale=1e14, a=-1, b=1, xmin=1e10, log=False):
    xt = x/scale
    xmint = xmin/scale
    if not log:
        d = b*(xt**a*np.exp(-xt**b))/(scale*gammainc((a + 1)/b, xmint**b))
    else:
        d = np.log(b) + a*np.log(xt) - xt**b - np.log(scale) - np.log(gammainc((a + 1)/b, xmint**b))
    return d


def _dtggd_log(x, scale=14, a=-1, b=1, xmin=10, log=False):
    xt = 10**(x - scale)
    xmint = 10**(xmin - scale)
    if not log:
        d = np.log(10)*b*(xt**(a + 1)*np.exp(-xt**b))/gammainc((a + 1)/b, xmint**b)
    else:
        d = np.log(ln10) + np.log(b) + (1 + a)*(x - scale)*ln10 - xt**b - np.log(gammainc((a + 1)/b, xmint**b))
    return d


def _dtggd_ln(x, scale=np.log(1e14), a=-1, b=1, xmin=np.log(1e10), log=False):
    xt = np.exp(x - scale)
    xmint = np.exp(xmin - scale)
    if not log:
        d = b*(xt**(a + 1)*np.exp(-xt**b))/gammainc((a + 1)/b, xmint**b)
    else:
        d = np.log(b) + (1 + a)*(x - scale) - xt**b - np.log(gammainc((a + 1)/b, xmint**b))
    return d


# ------------- CDFs --------------------------------------------
def _p_convert(p, lt, lg):
    if lt:
        if lg:
            p = np.log1p(-p)
        else:
            p = np.clip(1 - p,0,1)
    else:
        if lg:
            p = np.log(p)
    return p


def _ptggd(q, scale=1e14, a=-1, b=1, xmin=1e10, lower_tail=True, log_p=False):
    qt = q/scale
    xmint = xmin/scale
    p = gammainc((a + 1)/b, qt**b)/(gammainc((a + 1)/b, xmint**b))
    return _p_convert(p, lower_tail, log_p)


def _ptggd_log(q, scale=14, a=-1, b=1, xmin=10, lower_tail=True, log_p=False):
    qt = 10**(q - scale)
    xmint = 10**(xmin - scale)
    p = gammainc((a + 1)/b, qt**b)/gammainc((a + 1)/b, xmint**b)
    return _p_convert(p, lower_tail, log_p)


def _ptggd_ln(q, scale=np.log(1e14), a=-1, b=1, xmin=np.log(1e10), lower_tail=True, log_p=False):
    qt = np.exp(q - scale)
    xmint = np.exp(xmin - scale)
    p = gammainc((a + 1)/b, qt**b)/gammainc((a + 1)/b, xmint**b)
    return _p_convert(p, lower_tail, log_p)


# ------------- Quantiles --------------------------------------------
def _q_convert(p, lin_cdf, log_cdf, logm, log_p):
    lin_icdf = spline(lin_cdf, logm[:len(lin_cdf)])
    # the log-space icdf must be done in double log space.
    log_icdf = spline(np.log(-log_cdf)[::-1], logm[-len(log_cdf):][::-1])
    tp = lin_cdf[-1]
    if not log_p:
        log_pc = np.log(-np.log(np.clip(p[p > tp], None, 1)))
        lin_pc = np.clip(p[p <= tp], 0, None)
    else:
        log_pc = np.log(-np.clip(p[p > np.log(tp)], None, 0))
        lin_pc = np.exp(p[p <= np.log(tp)])

    if len(log_pc) and len(lin_pc) > 0:
        return np.concatenate((lin_icdf(lin_pc), log_icdf(log_pc)))
    elif len(lin_pc) > 0:
        return lin_icdf(lin_pc)
    else:
        return log_icdf(log_pc)


def _qtggd(p, scale=1e14, a=-1, b=1, xmin=1e10, lower_tail=True, log_p=False, res_approx=1e-2):
    mmax = scale*10**(2.5/b)
    logm = np.arange(np.log10(xmin), np.log10(mmax), res_approx)

    # Take a hybrid approach with log vals at high variates, and linear vals at low variates
    # First find a value at which to split between log and linear: around the transition scale
    # if that is above xmin.
    tp = max(np.log10(scale), np.log10(xmin) + 1)
    ind_tp = np.where(np.abs(tp - logm) == np.min(np.abs(tp - logm)))[0][0]

    # In the following, make sure the data overlaps on one index
    log_cdf = _ptggd_log(q=logm[ind_tp:], scale=np.log10(scale),
                         a=a, b=b, xmin=np.log10(xmin),
                         lower_tail=lower_tail, log_p=True)
    lin_cdf = _ptggd_log(q=logm[:(ind_tp + 1)], scale=np.log10(scale),
                         a=a, b=b, xmin=np.log10(xmin),
                         lower_tail=lower_tail)

    return 10**_q_convert(p, lin_cdf, log_cdf, logm, log_p)


def _qtggd_log(p, scale=14, a=-1, b=1, xmin=10, lower_tail=True, log_p=False, res_approx=1e-2):
    mmax = scale + 2.5/b
    logm = np.arange(xmin, mmax, res_approx)

    tp = max(scale, xmin+1)
    ind_tp = np.where(np.abs(tp - logm) == np.min(np.abs(tp - logm)))[0][0]

    log_cdf = _ptggd_log(q=logm[ind_tp:], scale=scale,
                         a=a, b=b, xmin=xmin,
                         lower_tail=lower_tail, log_p=True)
    lin_cdf = _ptggd_log(q=logm[:(ind_tp + 1)], scale=scale,
                         a=a, b=b, xmin=xmin,
                         lower_tail=lower_tail)

    return _q_convert(p, lin_cdf, log_cdf, logm, log_p)


def _qtggd_ln(p, scale=np.log(1e14), a=-1, b=1, xmin=np.log(1e10), lower_tail=True, log_p=False, res_approx=1e-2):
    mmax = scale + 2.5/b
    logm = np.arange(xmin/np.log(10), mmax/np.log(10), res_approx)

    tp = max(scale/np.log(10), (xmin+1)/np.log(10))
    ind_tp = np.where(np.abs(tp - logm) == np.min(np.abs(tp - logm)))[0][0]

    log_cdf = _ptggd_log(q=logm[ind_tp:], scale=scale/np.log(10),
                         a=a, b=b, xmin=xmin/np.log(10),
                         lower_tail=lower_tail, log_p=True)
    lin_cdf = _ptggd_log(q=logm[:(ind_tp + 1)], scale=scale/np.log(10),
                         a=a, b=b, xmin=xmin/np.log(10),
                         lower_tail=lower_tail)

    return _q_convert(p, lin_cdf, log_cdf, logm, log_p) * np.log(10)


# ------------- Random Variates --------------------------------------------
def _rv(func, n, scale, a, b, xmin, res_approx):
    return func(np.random.uniform(size=n), scale=scale, a=a, b=b, xmin=xmin, res_approx=res_approx)


def _rtggd(n, scale=1e14, a=-1, b=1, xmin=1e10, res_approx=1e-2):
    return _rv(_qtggd, n, scale, a, b, xmin, res_approx)


def _rtggd_log(n, scale=14, a=-1, b=1, xmin=10, res_approx=1e-2):
    return _rv(_qtggd_log, n, scale, a, b, xmin, res_approx)


def _rtggd_ln(n, scale=np.log(1e14), a=-1, b=1, xmin=np.log(1e10), res_approx=1e-2):
    return _rv(_qtggd_ln, n, scale, a, b, xmin, res_approx)

    # ----------------------- CLASSES ----------------------------------------


_init_par_doc = """
    Parameters
    ----------
    scale : array_like, optional
        Transition scale from power-law to exponential cut-off. Analogous to the scale
        parameter for the standard Gamma distribution.

    a : float or array_like, optional
        Power-law slope of the TGGD.

    b : float or array_like, optional
        Exponential cut-off parameter of the TGGD.

    xmin : float or array_like, optional
        Truncation value of the TGGD."""

_pdf_doc = """
    The pdf of the distribution.

    Parameters
    ----------
    x : array_like
        Variates at which to calculate the pdf. If any of `shape`, `a`, `b` or `xmin`
        are arrays, `x` must have the same length.

    log : logical, optional
        Whether to return the log of the pdf (if so, uses a better method than taking
        log of the final result).


    Returns
    -------
    d : float or array_like
        Values of the pdf corresponding to the variates `x`.
    """

_cdf_doc = """
    The cdf of the distribution.

    Parameters
    ----------
    q : array_like
        Variates at which to calculate the cdf. If any of `shape`, `a`, `b` or `xmin`
        are arrays, `q` must have the same length.

    lower_tail : logical, optional
        If `True` (default), probabilities are P[X <= q], otherwise, P[X > q].

    log_p : logical, optional
        If `True`, probabilities *p* are interpreated as log(*p*).


    Returns
    -------
    p : array_like
        The integrated probability of a variate being smaller than *q*.
    """

_quantile_doc = """
    The quantile of the distribution.

    The quantile at a given probability value *p* is defined as *q*, where

    .. math :: p = P(X \leq q).


    Parameters
    ----------
    p : array_like
        Probabilities at which to calculate the quantiles. If any of `shape`, `a`,
        `b` or `xmin` are arrays, `p` must have the same length.

    lower_tail : logical, optional
         If `True` (default), probabilities are P[X <= q], otherwise, P[X > q].

    log_p : logical, optional
        If `True`, probabilities *p* are returned as log(*p*).

    res_approx: float, optional
        Sets the resolution for interpolating the CDF, which is inverted to yield
        the quantile.


    Returns
    -------
    q : array_like
        The quantiles corresponding to *p*.
    """

_rvs_doc = """
    Generate random variates from the distribution.

    Parameters
    ----------
    n : integer or tuple of integers
        The size/shape of the returned variates. If an integer, specifies the number
        of returned variates. If a tuple of integers, the return array will have shape `n`.

    res_approx: float, optional
        Sets the resolution for interpolating the CDF, which is inverted to yield
        the quantile.

    Returns
    -------
    r : array_like
        Random variates from the distribution, with shape `n`.
    """

_mode_doc = """
    The mode of the distribution.
"""


class TGGD(object):
    def __init__(self, scale=1e14, a=-1.0, b=1.0, xmin=1e10):
        """
        The Truncated Generalised Gamma Distribution.

        The TGGD has the following PDF:

        .. math :: \frac{b (x/s)^a \exp(-(m/s)^b) }{\Gamma((a+1)/b,(xmin/s)^b)}

        where *s* corresponds to the scale argument of this class, and :math:`\Gamma`
        is the incomplete gamma function, provided by :package:`mpmath`.

        %(_init_par_doc)s

        Examples
        --------
        The following should create a sample and plot its histogram. The histogram
        should have a slope of -1.

        >>> from mrpy.stats import TGGD
        >>> import matplotlib.pyplot as plt
        >>> tggd = TGGD(a=-2)
        >>> r = tggd.rvs(100)
        >>> plt.hist(np.log10(r)) # doctest: +SKIP

        Taking the quantile of the cumulative probability at each variate should return
        something close to the variate.

        >>> a = tggd.quantile(tggd.cdf(r))/r  #should be close to 1
        >>> np.all(np.isclose(a,1))
        True

        Show that the numerical integral is equal to the CDF.

        >>> from scipy.integrate import quad
        >>> tggd = TGGD(scale=1e14,a=-1.5,b=0.7,xmin=1e10)
        >>> a = quad(tggd.pdf,1e10,1e11)[0]/tggd.cdf(1e11) # should be close to 1
        >>> np.isclose(a,1)
        True

        The CDF should approach unity when ``x >> scale``

        >>> a = tggd.cdf(1e18) #Should be close to 1
        >>> np.isclose(a,1)
        True

        To show the link to the ``log`` and ``ln`` variants, the following should be
        a sequence from 0 to 1 (by 0.1)

        >>> from mrpy.stats import TGGDlog, TGGDln
        >>> tggd = TGGD()
        >>> tggd_log = TGGDlog()
        >>> tggd_ln = TGGDln()
        >>> tggd.cdf(10**tggd_log.quantile(np.arange(0,1,0.1))) # doctest: +NORMALIZE_WHITESPACE
        array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])

        >>> b = tggd.cdf(np.exp(tggd_ln.quantile(np.arange(0,1,0.1))))
        >>> np.all(np.isclose(b,np.arange(0,1,0.1)))
        True
        """
        self.scale = scale
        self.a = a
        self.b = b
        self.xmin = xmin

    def pdf(self, x, log=False):
        '%(_pdf_doc)s'
        return _dtggd(x, self.scale, self.a, self.b, self.xmin, log)

    def cdf(self, q, lower_tail=True, log_p=False):
        '%(_cdf_doc)s'
        return _ptggd(q, self.scale, self.a, self.b, self.xmin, lower_tail, log_p)

    def quantile(self, p, lower_tail=True, log_p=False, res_approx=1e-2):
        '%(_quantile_doc)s'
        return _qtggd(p, self.scale, self.a, self.b, self.xmin, lower_tail, log_p, res_approx)

    def rvs(self, n, res_approx=1e-2):
        """
        %(_rvs_doc)s
        """
        return _rtggd(n, self.scale, self.a, self.b, self.xmin, res_approx)

    @property
    def mode(self):
        '%(_mode_doc)s'
        if self.a < 0:
            return self.xmin
        elif self.a == 0:
            return np.nan
        else:
            return self.scale*(self.a/self.b)**(1/self.b)

    @property
    def mean(self):
        return self.scale * (gammainc((self.a+2)/self.b,(self.xmin/self.scale)**self.b)/
                             gammainc((self.a+1)/self.b,(self.xmin/self.scale)**self.b))

class TGGDlog(object):
    def __init__(self, scale=14.0, a=-1.0, b=1.0, xmin=10.0):
        """
        The Truncated Generalised Gamma Distribution in log10 space.

        Specifically, if `10**x` is drawn from a TGGD distribution (in real space),
        this function gives the distribution of `x`, using the same parameter values.

        The log10 TGGD has the following PDF:

        .. math :: \frac{\ln(10) b (10^{(x-s)(a+1)} \exp(-10^{b(x-s)})}{s \Gamma(\frac{a+1}{b},10^{b(m-s)})}

        where *s* corresponds to the scale argument of this class, and :math:`\Gamma`
        is the incomplete gamma function, provided by :package:`mpmath`.

        %(_init_par_doc)s


        Examples
        --------
        The following should create a sample and plot its histogram. The histogram
        should have a slope of -1.

        >>> from mrpy.stats import TGGDlog
        >>> import matplotlib.pyplot as plt
        >>> tggdlog = TGGDlog(a=-2)
        >>> r = tggdlog.rvs(100)
        >>> plt.hist(r) # doctest: +SKIP

        Taking the quantile of the cumulative probability at each variate should return
        something close to the variate.

        >>> a = tggdlog.quantile(tggdlog.cdf(r))/r  #should be close to 1
        >>> np.all(np.isclose(a,1))
        True

        Show that the numerical integral is equal to the CDF.

        >>> from scipy.integrate import quad
        >>> tggdlog = TGGDlog(scale=14,a=-1.5,b=0.7,xmin=10)
        >>> a = quad(tggdlog.pdf,10,11)[0]/tggdlog.cdf(11) # should be close to 1
        >>> np.isclose(a,1)
        True

        The CDF should approach unity when ``x >> scale``

        >>> a = tggdlog.cdf(18) #Should be close to 1
        >>> np.isclose(a,1)
        True

        To show the link to the ``log`` and ``ln`` variants, the following should be
        a sequence from 0 to 1 (by 0.1)

        >>> from mrpy.stats import TGGDlog, TGGDln
        >>> tggd = TGGD()
        >>> tggd_log = TGGDlog()
        >>> tggd_ln = TGGDln()
        >>> tggd_ln.cdf(np.log(tggd.quantile(np.arange(0,1,0.1)))) # doctest: +NORMALIZE_WHITESPACE
        array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])

        >>> b = tggd_ln.cdf(tggd_log.quantile(np.arange(0,1,0.1))*np.log(10))
        >>> np.all(np.isclose(b,np.arange(0,1,0.1)))
        True
        """
        self.scale = scale
        self.a = a
        self.b = b
        self.xmin = xmin

    def pdf(self, x, log=False):
        '%(_pdf_doc)s'
        return _dtggd_log(x, self.scale, self.a, self.b, self.xmin, log)

    def cdf(self, q, lower_tail=True, log_p=False):
        '%(_cdf_doc)s'
        return _ptggd_log(q, self.scale, self.a, self.b, self.xmin, lower_tail, log_p)

    def quantile(self, p, lower_tail=True, log_p=False, res_approx=1e-2):
        '%(_quantile_doc)s'
        return _qtggd_log(p, self.scale, self.a, self.b, self.xmin, lower_tail, log_p, res_approx)

    def rvs(self, n, res_approx=1e-2):
        '%(_rvs_doc)s'
        return _rtggd_log(n, self.scale, self.a, self.b, self.xmin, res_approx)

    @property
    def mode(self):
        '%(_mode_doc)s'
        if self.a < -1:
            return self.xmin
        elif self.a == -1:
            return np.nan
        else:
            return self.scale + ((self.a + 1)/self.b)**(1/self.b)


class TGGDln(object):
    def __init__(self, scale=np.log(1e14), a=-1.0, b=1.0, xmin=np.log(1e10)):
        """
        The Truncated Generalised Gamma Distribution in ln space.

        Specifically, if `exp(x)` is drawn from a TGGD distribution (in real space),
        this function gives the distribution of `x`, using the same parameter values.

        The ln TGGD has the following PDF:

        .. math :: \frac{ b (\exp((x-s)(a+1)) \exp(-\exp(b(x-s)))}{s \Gamma(\frac{a+1}{b},\exp(b(x-s)))}

        where *s* corresponds to the scale argument of this class, and :math:`\Gamma`
        is the incomplete gamma function, provided by :package:`mpmath`.

        %(_init_par_doc)s


        Examples
        --------
        The following should create a sample and plot its histogram. The histogram
        should have a slope of -1.

        >>> from mrpy.stats import TGGDln
        >>> import matplotlib.pyplot as plt
        >>> tggdln = TGGDln(a=-2)
        >>> r = tggdln.rvs(100)
        >>> plt.hist(r) # doctest: +SKIP

        Taking the quantile of the cumulative probability at each variate should return
        something close to the variate.

        >>> a = tggdln.quantile(tggdln.cdf(r))/r  #should be close to 1
        >>> np.all(np.isclose(a,1))
        True

        Show that the numerical integral is equal to the CDF.

        >>> from scipy.integrate import quad
        >>> tggdln = TGGDln(scale=14,a=-1.5,b=0.7,xmin=10)
        >>> a = quad(tggdln.pdf,10,11)[0]/tggdln.cdf(11) # should be close to 1
        >>> np.isclose(a,1)
        True

        The CDF should approach unity when ``x >> scale``

        >>> a = tggdln.cdf(np.log(1e18)) #Should be close to 1
        >>> np.isclose(a,1)
        True

        To show the link to the ``log`` and ``ln`` variants, the following should be
        a sequence from 0 to 1 (by 0.1)

        >>> from mrpy.stats import TGGDlog, TGGDln
        >>> tggd = TGGD()
        >>> tggd_log = TGGDlog()
        >>> tggd_ln = TGGDln()
        >>> tggd_log.cdf(np.log10(tggd.quantile(np.arange(0,1,0.1)))) # doctest: +NORMALIZE_WHITESPACE
        array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9])

        >>> b = tggd_log.cdf(tggd_ln.quantile(np.arange(0,1,0.1))/np.log(10))
        >>> np.all(np.isclose(b,np.arange(0,1,0.1)))
        True
        """
        self.scale = scale
        self.a = a
        self.b = b
        self.xmin = xmin

    def pdf(self, x, log=False):
        '%(_pdf_doc)s'
        return _dtggd_ln(x, self.scale, self.a, self.b, self.xmin, log)

    def cdf(self, q, lower_tail=True, log_p=False):
        '%(_cdf_doc)s'
        return _ptggd_ln(q, self.scale, self.a, self.b, self.xmin, lower_tail, log_p)

    def quantile(self, p, lower_tail=True, log_p=False, res_approx=1e-2):
        '%(_quantile_doc)s'
        return _qtggd_ln(p, self.scale, self.a, self.b, self.xmin, lower_tail, log_p, res_approx)

    def rvs(self, n, res_approx=1e-2):
        '%(_rvs_doc)s'
        return _rtggd_ln(n, self.scale, self.a, self.b, self.xmin, res_approx)

    def mode(self):
        '%(_mode_doc)s'
        if self.a < -1:
            return self.xmin
        elif self.a == -1:
            return np.nan
        else:
            return self.scale + ((self.a + 1)/self.b)**(1/self.b)
