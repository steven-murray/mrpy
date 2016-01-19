"""
A module defining the TGGD distribution (as well as in log and ln space) in standard R style.

See the R package `tggd` for a similar implementation.

The distribution functions implemented here are described in detail in Murray, Robotham and Power, 2016.

.. note:: *Which distribution should I choose?* The log/ln versions in this module are intended to provide the correct
          distribution when variates are drawn from a real-space TGGD, but there are priors on their uncertainty which
          operate in log-space (eg. a log-normal distribution). The likelihood of a given set of parameters is incorrect
          in such a case if the real-space version is used without an adjustment to the Jacobian.

          Short answer: generally use :class:`TGGD`, but if your variates must form a proper PDF in *log-space*, use the
          appropriate log-space version.
"""

import numpy as np
import special as sp
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy.misc import comb as _comb
import _utils

ln10 = np.log(10)

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


class TGGD(object):
    r"""
    The Truncated Generalised Gamma Distribution.

    The TGGD has the following PDF:

    .. math:: f(x) = \frac{b \left(\frac{x}{s}\right)^a \exp\left(-\left(\frac{m}{s}\right)^b\right)}{\Gamma\left(\frac{a+1}{b},\left(\frac{x_{\rm min}}{s}\right)^b\right)}

    where :math:`s>0` corresponds to the scale argument of this class, :math:`a \in \Re`, :math:`b>0`, :math:`x_{\rm min}>0`
    is the truncation value, \and :math:`\Gamma` is the incomplete gamma function, provided by `mpmath`.

    %s

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

    Show that the numerical integral is equal to the CDF. The following uses a `scale`
    more appropriate to halo mass functions.

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
    __doc__ %= _init_par_doc

    def __init__(self, scale=1, a=-1.0, b=1.0, xmin=0.1):

        self.scale = scale
        self.a = a
        self.b = b
        self.xmin = xmin

    # =====================================================================================
    # Private Methods/Attributes
    # =====================================================================================
    @property
    def _xmint(self):
        return self.xmin/self.scale

    @property
    def _xmintb(self):
        return self._xmint**self.b

    def _xt(self, x):
        return x/self.scale

    @property
    def _z(self):
        return (self.a + 1)/self.b

    def _pdf_shape(self, x, log=False):
        xt = self._xt(x)
        if not log:
            return self.b*xt**self.a*np.exp(-xt**self.b)
        else:
            return np.log(self.b) + self.a*np.log(xt) - xt**self.b

    def _pdf_norm(self, log=False):
        a = self.scale*sp.gammainc(self._z, self._xmintb)
        if not log:
            return a
        else:
            return np.log(a)

    def _cdf_convert(self, p, lt, lg):
        if lt:
            if lg:
                p = np.log1p(-p)
            else:
                p = np.clip(1 - p, 0, 1)
        else:
            if lg:
                p = np.log(p)
        return p

    def _q_convert(self, p, lin_cdf, log_cdf, logm, log_p):
        lin_icdf = _spline(lin_cdf, logm[:len(lin_cdf)])
        # the log-space icdf must be done in double log space.
        log_icdf = _spline(np.log(-log_cdf)[::-1], logm[-len(log_cdf):][::-1])
        tp = lin_cdf[-1]
        if np.isscalar(p):
            p = np.array(p)
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

    @property
    def _q_xmax(self):
        return self.scale*10**(2.5/self.b)

    # =====================================================================================
    # Public Methods/Attributes
    # =====================================================================================
    def pdf(self, x, log=False):
        """
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
        if log:
            return self._pdf_shape(x, log) - self._pdf_norm(log)
        else:
            return self._pdf_shape(x, log)/self._pdf_norm(log)

    def cdf(self, q, lower_tail=True, log_p=False):
        """
        The cdf of the distribution.

        Parameters
        ----------
        q : array_like
            Variates at which to calculate the cdf. If any of `shape`, `a`, `b` or `xmin`
            are arrays, `q` must have the same length.

        lower_tail : logical, optional
            If `True` (default), probabilities are P[X <= q], otherwise, P[X > q].

        log_p : logical, optional
            If `True`, probabilities *p* are interpreted as log(*p*).


        Returns
        -------
        p : array_like
            The integrated probability of a variate being smaller than *q*.
        """
        qt = self._xt(q)
        p = sp.gammainc(self._z, qt**self.b)/sp.gammainc(self._z, self._xmintb)
        return self._cdf_convert(p, lower_tail, log_p)

    def quantile(self, p, lower_tail=True, log_p=False, res_approx=1e-2):
        """
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
        xmax = self._q_xmax
        logx = np.arange(np.log10(self.xmin), np.log10(xmax), res_approx)

        # Take a hybrid approach with log vals at high variates, and linear vals at low variates
        # First find a value at which to split between log and linear: around the transition scale
        # if that is above xmin.
        tp = max(np.log10(self.scale), np.log10(self.xmin) + 1)
        ind_tp = np.where(np.abs(tp - logx) == np.min(np.abs(tp - logx)))[0][0]

        # In the following, make sure the data overlaps on one index
        tggdlog = TGGDlog(scale=np.log10(self.scale), a=self.a, b=self.b, xmin=np.log10(self.xmin))

        log_cdf = tggdlog.cdf(q=logx[ind_tp:], lower_tail=lower_tail, log_p=True)
        # log_cdf = _ptggd_log(q=logm[ind_tp:], scale=np.log10(self.scale),
        #                      a=a, b=b, xmin=np.log10(xmin),
        #                      lower_tail=lower_tail, log_p=True)

        lin_cdf = tggdlog.cdf(q=logx[:(ind_tp + 1)], lower_tail=lower_tail)

        # lin_cdf = _ptggd_log(q=logm[:(ind_tp + 1)], scale=np.log10(scale),
        #                      a=a, b=b, xmin=np.log10(xmin),
        #                      lower_tail=lower_tail)

        return 10**self._q_convert(p, lin_cdf, log_cdf, logx, log_p)

    def rvs(self, n, res_approx=1e-2):
        """
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
        return self.quantile(np.random.uniform(size=n), res_approx=res_approx)

    @property
    def mode(self):
        """
        The mode of the distribution
        """
        if self.a < 0:
            return self.xmin
        elif self.a == 0:
            return np.nan
        else:
            return self.scale*(self.a/self.b)**(1/self.b)

    @property
    def mean(self):
        """
        The mean of the distribution.

        Examples
        --------
        >>> from mrpy.stats import TGGD
        >>> t = TGGD()
        >>> r = t.rvs(1e6)
        >>> np.isclose(np.mean(r)/t.mean,1,rtol=1e-2) #should be close to 1
        True
        """
        return self.raw_moments(1)

    def raw_moments(self, n):
        """
        Calculate the nth raw moment, E[X^n].

        Parameters
        ----------
        n : array_like
            The order(s) of the moment desired.

        Returns
        -------
        mu_n : array_like
            The raw moment(s) corresponding to the order(s) `n`.

        Notes
        -----
        The 1st raw moment is equivalent to the mean.

        Examples
        --------
        >>> from mrpy.stats import TGGD
        >>> t = TGGD()
        >>> mean = t.raw_moments(1)
        >>> ten_moments = t.raw_moments(np.arange(10)) #doctest: +SKIP
        """
        zn = (self.a + 1 + n)/self.b
        z0 = (self.a + 1)/self.b
        x = (self.xmin/self.scale)**self.b
        return self.scale**n*sp.gammainc(zn, x)/sp.gammainc(z0, x)

    def central_moments(self, n):
        """
        Calculate the nth central moment, E[(X-mu)^n].

        Parameters
        ----------
        n : integer
            The order of the moment desired.

        Returns
        -------
        mu_n : float
            The nth central moment.

        Notes
        -----
        The 2nd central moment is equivalent to the variance.

        Examples
        --------
        >>> from mrpy.stats import TGGD
        >>> t = TGGD()
        >>> variance = t.central_moments(2)
        >>> t.raw_moments(10)
        199064.8037313875
        """
        k = np.arange(n + 1)
        sign = (-1)**(n - k)

        zk = (self.a + 1 + k)/self.b
        z0 = (self.a + 1)/self.b
        z1 = (self.a + 2)/self.b

        x = (self.xmin/self.scale)**self.b

        coeffs = _comb(n, k)

        return self.scale**n*np.sum(coeffs*sp.gammainc(z1, x)**(n - k)*
                                    sp.gammainc(zk, x)*sign/sp.gammainc(z0, x)**(n - k + 1))

    @property
    def variance(self):
        """
        The variance of the distribution.

        Examples
        --------
        >>> from mrpy.stats import TGGD
        >>> t = TGGD()
        >>> r = t.rvs(1e6)
        >>> np.isclose(np.var(r)/t.variance,1,rtol=1e-1) # should be close to 1
        True
        """
        return self.central_moments(2)

    def normalised_central_moments(self, n):
        """
        Calculate the nth standardized central moment, E[(X-mu)^n]/sigma^n.

        Parameters
        ----------
        n : integer
            The order of the moment desired.

        Returns
        -------
        mu_n : float
            The nth standardized central moment.

        Notes
        -----
        The 3rd standardized central moment is equivalent to the skewness.
        """
        return self.central_moments(n)/self.variance**(n/2.0)

    @property
    def skewness(self):
        """
        The 3rd-order moment of the distribution.

        Examples
        --------
        >>> from mrpy.stats import TGGD
        >>> from scipy.stats import skew
        >>> t = TGGD()
        >>> r = t.rvs(1e6)
        >>> np.isclose(skew(r)/t.skewness,1,rtol=1e-1) #should be close to 1
        True
        """
        return self.normalised_central_moments(3)

    @property
    def kurtosis(self):
        """
        The 4th-order moment of the distribution

        Examples
        --------
        >>> from mrpy.stats import TGGD
        >>> from scipy.stats import kurtosis
        >>> t = TGGD()
        >>> r = t.rvs(1e6)
        >>> np.isclose(kurtosis(r)/t.kurtosis,1,rtol=1e-1) #should be close to 1
        True
        """
        return self.normalised_central_moments(4) - 3.0

    @property
    def hyperskewness(self):
        """
        The 5th-order moment of the distribution
        """
        return self.normalised_central_moments(5)

    @property
    def hyperflatness(self):
        """
        The 6th order moment of the distribution
        """
        return self.normalised_central_moments(6)


class TGGDlog(TGGD):
    r"""
    The Truncated Generalised Gamma Distribution in log10 space.

    Specifically, if `10**x` is drawn from a TGGD distribution (in real space),
    this function gives the distribution of `x`, using the same parameter values.

    The log10 TGGD has the following PDF:

    .. math :: \frac{\ln(10) b (10^{(x-s)(a+1)} \exp(-10^{b(x-s)})}{s \Gamma(\frac{a+1}{b},10^{b(m-s)})}

    where *s* corresponds to the scale argument of this class, and :math:`\Gamma`
    is the incomplete gamma function, provided by `mpmath`.

    %s


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

    Show that the numerical integral is equal to the CDF. We use a `scale` appropriate for an example such as a halo
    mass function.

    >>> from scipy.integrate import quad
    >>> tggdlog = TGGDlog(scale=14,a=-1.5,b=0.7,xmin=10)
    >>> a = quad(tggdlog.pdf,10,11)[0]/tggdlog.cdf(11) # should be close to 1
    >>> np.isclose(a,1)
    True

    The CDF should approach unity when ``10**x >> 10**scale``

    >>> a = tggdlog.cdf(18) #Should be close to 1
    >>> np.isclose(a,1)
    True

    To show the link to the ``log`` and ``ln`` variants, the following should be
    a sequence from 0 to 1 (by 0.1)

    >>> from mrpy.stats import TGGDlog, TGGDln
    >>> tggd = TGGD()
    >>> tggd_log = TGGDlog()
    >>> tggd_ln = TGGDln()
    >>> a = tggd_ln.cdf(np.log(tggd.quantile(np.arange(0,1,0.1)))) # doctest: +NORMALIZE_WHITESPACE
    >>> np.all(np.isclose(a,np.arange(0,1,0.1)))
    True

    >>> b = tggd_ln.cdf(tggd_log.quantile(np.arange(0,1,0.1))*np.log(10))
    >>> np.all(np.isclose(b,np.arange(0,1,0.1)))
    True
    """
    __doc__ %= _init_par_doc

    def __init__(self, scale=0.0, a=-1.0, b=1.0, xmin=-1.0):
        super(TGGDlog, self).__init__(scale, a, b, xmin)

    # =====================================================================================
    # Private Methods/Attributes
    # =====================================================================================
    def _xt(self,x):
        return 10**(x-self.scale)

    @property
    def _xmint(self):
        return 10**(self.xmin - self.scale)

    def _pdf_shape(self, x, log=False):
        xt = 10**(x - self.scale)
        if not log:
            return ln10*self.b*xt**(self.a + 1)*np.exp(-xt**self.b)
        else:
            return np.log(ln10*self.b) + (self.a + 1)*np.log(xt) - xt**self.b

    def _pdf_norm(self, log=False):
        a = sp.gammainc(self._z, self._xmintb)
        if not log:
            return a
        else:
            return np.log(a)

    @property
    def _q_xmax(self):
        return self.scale + (2.5/self.b)

    # =====================================================================================
    # Public Methods/Attributes
    # =====================================================================================
    @_utils.copydoc(TGGD.quantile)
    def quantile(self, p, lower_tail=True, log_p=False, res_approx=1e-2):
        xmax = self._q_xmax
        logx = np.arange(self.xmin, xmax, res_approx)

        tp = max(self.scale, self.xmin + 1)
        ind_tp = np.where(np.abs(tp - logx) == np.min(np.abs(tp - logx)))[0][0]

        log_cdf = self.cdf(q=logx[ind_tp:], lower_tail=lower_tail, log_p=True)
        lin_cdf = self.cdf(q=logx[:(ind_tp + 1)], lower_tail=lower_tail)
        return self._q_convert(p, lin_cdf,log_cdf,logx, log_p)

    @property
    def mode(self):
        """
        The mode of the distribution
        """
        if self.a < -1:
            return self.xmin
        elif self.a == -1:
            return np.nan
        else:
            return self.scale + self._z**(1/self.b)


class TGGDln(TGGDlog):
    r"""
    The Truncated Generalised Gamma Distribution in ln space.

    Specifically, if `exp(x)` is drawn from a TGGD distribution (in real space),
    this function gives the distribution of `x`, using the same parameter values.

    The ln TGGD has the following PDF:

    .. math :: \frac{ b (\exp((x-s)(a+1)) \exp(-\exp(b(x-s)))}{s \Gamma(\frac{a+1}{b},\exp(b(x-s)))}

    where *s* corresponds to the scale argument of this class, and :math:`\Gamma`
    is the incomplete gamma function, provided by `mpmath`.

    %s

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
    >>> a = tggd_log.cdf(np.log10(tggd.quantile(np.arange(0,1,0.1)))) # doctest: +NORMALIZE_WHITESPACE
    >>> np.all(np.isclose(a,np.arange(0,1,0.1)))
    True

    >>> a = tggd_log.cdf(tggd_ln.quantile(np.arange(0,1,0.1))/np.log(10))
    >>> np.all(np.isclose(a,np.arange(0,1,0.1)))
    True
    """
    __doc__ %= _init_par_doc

    def __init__(self, scale=0.0, a=-1.0, b=1.0, xmin=-1.0*ln10):
        super(TGGDln, self).__init__(scale, a, b, xmin)

    # =====================================================================================
    # Private Methods/Attributes
    # =====================================================================================
    def _xt(self,x):
        return np.exp(x-self.scale)

    @property
    def _xmint(self):
        return np.exp(self.xmin - self.scale)

    def _pdf_shape(self, x, log=False):
        xt = np.exp(x - self.scale)
        if not log:
            return self.b*xt**(self.a + 1)*np.exp(-xt**self.b)
        else:
            return np.log(self.b) + (self.a + 1)*np.log(xt) - xt**self.b