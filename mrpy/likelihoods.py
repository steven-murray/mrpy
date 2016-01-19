"""
Provides classes which extend the basic :class:`mrpy.core.MRP` class.

Adds methods for calculating the likelihood and its derivatives in special cases
of interest. Specifically, the two main cases of interest are fitting the MRP
per-object (PO) or fitting to a binned (or theoretical) curve.

At this time, we don't directly support fitting MRP extensions, such as a double-MRP.
"""
import special as sp
import core
import numpy as np
import scipy.integrate as intg
from cached_property import cached_property as _cached
import stats

ln10 = np.log(10)


class PerObjLike(core.MRP):
    """
    A subclass of :class:`mrpy.core.MRP` which adds the likelihood (and derivatives)
    of a model given data in the form of individual masses.

    In this case, the fitting process can be generalised to be weighted by some
    power of the mass, which may improve fitted quality. We include this power,
    s, in this class. This scaling doesn't touch the standard MRP outputs, just
    the quantities involved in fitting introduced here.

    Parameters
    ----------
    logm : array_like
        Vector of log10 masses.

    logHs, alpha, beta : array_like
        The shape parameters of the MRP.

    scale : float, optional
        Mass scale with which to weight the individual masses. This can
        be useful to gain sensitivity to the high-mass haloes.

    norm : float or string
        Gives the normalisation of the MRP, *A*. If set to a *float*, it
        is directly the normalisation. If set to ``"pdf"``, it will automatically
        render the MRP as a statistical distribution. If set to ``"rhoc"``, it will
        yield the correct total mass density across all masses, down to ``m=0``.

    log_mmin : array_like, optional
        Log-10 truncation mass of the MRP. By default is set to the minimum mass
        in ``logm``.

    Om0 : float, optional
        Matter density of the Universe. Only required if `norm` is set to ``Arhoc``.

    rhoc : float, optional
        Critical density of the Universe. Only required if `norm` is set to ``Arhoc``.
    """

    # Internally, some of the properties are defined twice -- once for the
    # truncation mass, and once for all masses in logm. For example, ``x`` is
    # :math:`(m/H_s)^{\beta}, but for various purposes, ``m`` could be the full array
    # or just the truncation mass. Throughout, quantities that are defined as the
    # truncation mass have an extra trailing underscore in their name.

    def __init__(self, logm, logHs, alpha, beta, scale=0, norm="pdf", log_mmin=None,
                 Om0=0.3, rhoc=2.7755e11):
        self.scale = scale
        super(PerObjLike, self).__init__(logm, logHs, alpha, beta, norm, log_mmin, Om0, rhoc)
        self._alpha_s = self.alpha + scale

    # ===========================================================================
    # Basic unit quantities
    # ===========================================================================
    @_cached
    def _z(self):
        """
        The term z = (1+alpha)/beta.
        """
        return (self._alpha_s + 1)/self.beta

    @_cached
    def _y_(self):
        """
        The scaled truncation mass
        """
        return self.mmin/self.Hs

    @_cached
    def _x_(self):
        """
        y^beta (truncation mass scaled)
        """
        return self._y_**self.beta

    @_cached
    def _y(self):
        """
        The scaled masses
        """
        return self.m/self.Hs

    @_cached
    def _x(self):
        """
        y^beta (all masses, not just truncation)
        """
        return self._y**self.beta

    # ===========================================================================
    # Cached special functions
    # ===========================================================================
    @_cached
    def _gammaz(self):
        """
        The gamma function at z=(1+a)/b. Stored for use elsewhere.
        """
        return sp.gamma(self._z)

    @_cached
    def _gammainc_zx(self):
        """
        The incomplete gamma function, Gamma(z,x), where z,x are as specified in
        this class.
        """
        return sp.gammainc(self._z, self._x)

    @_cached
    def _gammainc_zx_(self):
        """
        The incomplete gamma function, Gamma(z,x), where z,x are as specified in
        this class.
        """
        return sp.gammainc(self._z, self._x_)

    @_cached
    def _G1(self):
        return sp.G1(self._z, self._x)

    @_cached
    def _G1_(self):
        return sp.G1(self._z, self._x_)

    @_cached
    def _G2(self):
        return sp.G2(self._z, self._x)

    @_cached
    def _G2_(self):
        return sp.G2(self._z, self._x_)

    @_cached
    def _Gbar(self):
        return self._G1*np.log(self._x) + 2*self._G2

    @_cached
    def _Gbar_(self):
        return self._G1_*np.log(self._x_) + 2*self._G2_

    @_cached
    def _phi(self):
        return self._y*self._g/self._gammainc_zx

    @_cached
    def _phi_(self):
        return self._y_*self._g_/self._gammainc_zx_

    # ===========================================================================
    # Mass scaling utilities
    # ===========================================================================
    @_cached
    def _scaled_mass(self):
        return self.m**self.scale

    @_cached
    def _mean_scaling(self):
        return np.mean(self._scaled_mass)

    # ===========================================================================
    # Basic MRP quantities, renamed for compactness
    # ===========================================================================
    @_cached
    def _g(self):
        """
        The shape of the MRP, completely unnormalised (ie. A=1) (all masses)
        """
        return core.dndm(self.m, self.logHs, self._alpha_s, self.beta, self.mmin, norm=1)

    @_cached
    def _g_(self):
        """
        The shape of the MRP, completely unnormalised (ie. A=1) (truncation mass)
        """
        return core.dndm(self.mmin, self.logHs, self._alpha_s, self.beta, self.mmin,norm=1)

    @_cached
    def _lng(self):
        """
        Better log of g than log(g) (all masses)
        """
        return core.dndm(self.m, self.logHs, self._alpha_s, self.beta, norm=1, log=True)

    @_cached
    def _lng_(self):
        """
        Better log of g than log(g) (truncation mass)
        """
        return core.dndm(self.mmin, self.logHs, self._alpha_s, self.beta, self.mmin,norm=1, log=True)

    @_cached
    def _q(self):
        """
        The normalisation of the MRP (ie. integral of g) (all masses)
        """
        return stats.TGGD(scale=self.Hs, a=self._alpha_s, b=self.beta, xmin=self.m)._pdf_norm()

    @_cached
    def _q_(self):
        """
        The normalisation of the MRP (ie. integral of g) (truncation masses)
        """
        return stats.TGGD(scale=self.Hs, a=self._alpha_s, b=self.beta, xmin=self.mmin)._pdf_norm()

    @_cached
    def _lnq(self):
        """
        The log normalisation of the MRP (ie. integral of g) (all masses)
        """
        return np.log(self._q)

    @_cached
    def _lnq_(self):
        """
        The normalisation of the MRP (ie. integral of g) (truncation masses)
        """
        return np.log(self._q_)

    # ===========================================================================
    # Basic likelihood
    # ===========================================================================
    @_cached
    def _lnLi(self):
        """
        Logarithmic likelihood of the particles, given its mass and a model (uniform prior)
        """
        return self._scaled_mass*(self._lng - np.log(self._q_))/self._mean_scaling

    @property
    def lnL(self):
        """
        Total log-likelihood with current model for masses m [uniform prior]
        """
        return np.sum(self._lnLi)

    # ===========================================================================
    # Simple Derivatives
    # ===========================================================================
    # ----------- g'() --------------------------------------
    @_cached
    def _lng_h(self):
        return (self.beta*self._x - self._alpha_s)*ln10

    @_cached
    def _lng_h_(self):
        return (self.beta*self._x_ - self._alpha_s)*ln10

    @_cached
    def _lng_a(self):
        return np.log(self._y)

    @_cached
    def _lng_a_(self):
        return np.log(self._y_)

    @_cached
    def _lng_b(self):
        return (1 - self._x*np.log(self._x))/self.beta

    @_cached
    def _lng_b_(self):
        return (1 - self._x_*np.log(self._x_))/self.beta

    @_cached
    def _lng_h_h_(self):
        return -ln10**2*self.beta**2*self._x_

    @_cached
    def _lng_h_h(self):
        return -ln10**2*self.beta**2*self._x

    @_cached
    def _lng_a_a_(self):
        return 0

    @_cached
    def _lng_a_a(self):
        return np.zeros(len(self.m))

    @_cached
    def _lng_a_b_(self):
        return 0

    @_cached
    def _lng_a_b(self):
        return np.zeros(len(self.m))

    @_cached
    def _lng_b_b_(self):
        return -1/self.beta**2 - self._x_*np.log(self._y_)**2

    @_cached
    def _lng_b_b(self):
        return -1/self.beta**2 - self._x*np.log(self._y)**2

    @_cached
    def _lng_h_a_(self):
        return -ln10

    @_cached
    def _lng_h_a(self):
        return -np.ones(len(self.m))*ln10

    @_cached
    def _lng_h_b_(self):
        return self._x_*(1 + np.log(self._x_))*ln10

    @_cached
    def _lng_h_b(self):
        return self._x*(1 + np.log(self._x))*ln10

    @_cached
    def _lng_jac(self):
        return np.array([getattr(self, "_lng_%s"%x) for x in 'hab'])

    @_cached
    def _lng_hess(self):
        return np.array([getattr(self, "_lng_%s_%s"%(x, y)) for x, y in ('hh', 'ha', 'hb',
                                                                         'ha', 'aa', 'ab',
                                                                         'hb', 'ab', 'bb')]).reshape(
                (3, 3, len(self.m)))

    @_cached
    def _lng_jac_(self):
        return np.array([getattr(self, "_lng_%s_"%x) for x in 'hab'])

    @_cached
    def _lng_hess_(self):
        return np.array([getattr(self, "_lng_%s_%s_"%(x, y)) for x, y in ('hh', 'ha', 'hb',
                                                                          'ha', 'aa', 'ab',
                                                                          'hb', 'ab', 'bb')]).reshape((3, 3))

    # --------- \tilde{q}'() --------------------------------------
    @_cached
    def _lnq_h(self):
        return ln10*(1 + self._phi)

    @_cached
    def _lnq_h_(self):
        return ln10*(1 + self._phi_)

    @_cached
    def _lnq_a(self):
        return (np.log(self._x) + self._G1)/self.beta

    @_cached
    def _lnq_a_(self):
        return (np.log(self._x_) + self._G1_)/self.beta

    @_cached
    def _lnq_b(self):
        return -self._z*self._lnq_a - self._phi*np.log(self._y)/self.beta

    @_cached
    def _lnq_b_(self):
        return -self._z*self._lnq_a_ - self._phi_*np.log(self._y_)/self.beta

    @_cached
    def _q_h_h(self):
        return ln10*(self._lnq_h + self._phi*self._lng_h)

    @_cached
    def _q_h_h_(self):
        return ln10*(self._lnq_h_ + self._phi_*self._lng_h_)

    @_cached
    def _lnq_h_h(self):
        return self._q_h_h - self._lnq_h**2

    @_cached
    def _lnq_h_h_(self):
        return self._q_h_h_ - self._lnq_h_**2

    @_cached
    def _q_a_a(self):
        return (1.0/self.beta)*(self._lnq_a*np.log(self._x) + self._Gbar/self.beta)

    @_cached
    def _q_a_a_(self):
        return (1.0/self.beta)*(self._lnq_a_*np.log(self._x_) + self._Gbar_/self.beta)

    @_cached
    def _lnq_a_a(self):
        return self._q_a_a - self._lnq_a**2

    @_cached
    def _lnq_a_a_(self):
        return self._q_a_a_ - self._lnq_a_**2

    @_cached
    def _q_a_b(self):
        b = self.beta
        lnx = np.log(self._x)
        return (1.0/self.beta)*(-self._lnq_a + self._lnq_b*lnx - (self._z*self._Gbar/b))

    @_cached
    def _q_a_b_(self):
        b = self.beta
        lnx = np.log(self._x_)
        return (1.0/self.beta)*(-self._lnq_a_ + self._lnq_b_*lnx - (self._z*self._Gbar_/b))

    @_cached
    def _lnq_a_b(self):
        return self._q_a_b - self._lnq_a*self._lnq_b

    @_cached
    def _lnq_a_b_(self):
        return self._q_a_b_ - self._lnq_a_*self._lnq_b_

    @_cached
    def _q_b_b(self):
        b = self.beta
        return (self._phi*np.log(self._y)/b)*(1./b - self._lng_b) - self._z*(self._q_a_b - self._lnq_a/b)

    @_cached
    def _q_b_b_(self):
        b = self.beta
        return (self._phi_*np.log(self._y_)/b)*(1./b - self._lng_b_) - self._z*(self._q_a_b_ - self._lnq_a_/b)

    @_cached
    def _lnq_b_b(self):
        return self._q_b_b - self._lnq_b*self._lnq_b

    @_cached
    def _lnq_b_b_(self):
        return self._q_b_b_ - self._lnq_b_*self._lnq_b_

    @_cached
    def _q_h_a(self):
        return ln10*(self._lnq_a + self._phi*self._lng_a)

    @_cached
    def _q_h_a_(self):
        return ln10*(self._lnq_a_ + self._phi_*self._lng_a_)

    @_cached
    def _lnq_h_a(self):
        return self._q_h_a - self._lnq_a*self._lnq_h

    @_cached
    def _lnq_h_a_(self):
        return self._q_h_a_ - self._lnq_a_*self._lnq_h_

    @_cached
    def _q_h_b(self):
        return ln10*(self._lnq_b + self._phi*self._lng_b)

    @_cached
    def _q_h_b_(self):
        return ln10*(self._lnq_b_ + self._phi_*self._lng_b_)

    @_cached
    def _lnq_h_b(self):
        return self._q_h_b - self._lnq_h*self._lnq_b

    @_cached
    def _lnq_h_b_(self):
        return self._q_h_b_ - self._lnq_h_*self._lnq_b_

    @_cached
    def _lnq_jac(self):
        return np.array([getattr(self, "_lnq_%s"%x) for x in 'hab'])

    @_cached
    def _lnq_jac_(self):
        return np.array([getattr(self, "_lnq_%s_"%x) for x in 'hab'])

    @_cached
    def _lnq_hess(self):
        return np.array([getattr(self, "_lnq_%s_%s"%(x, y)) for x, y in ('hh', 'ha', 'hb',
                                                                         'ha', 'aa', 'ab',
                                                                         'hb', 'ab', 'bb')]).reshape(
                (3, 3, len(self.m)))

    @_cached
    def _lnq_hess_(self):
        return np.array([getattr(self, "_lnq_%s_%s_"%(x, y)) for x, y in ('hh', 'ha', 'hb',
                                                                          'ha', 'aa', 'ab',
                                                                          'hb', 'ab', 'bb')]).reshape((3, 3))

    # ------ Matrices ------------------
    def _Q_x_y(self, x, y):
        """
        Double derivatives of Q w.r.t x then y
        """
        if y == "h" or (y == "a" and x == "b"):
            x, y = y, x

        lngxy = getattr(self, "_lng_%s_%s"%(x, y))
        lnqxy = getattr(self, "_lnq_%s_%s_"%(x, y))

        return lngxy - lnqxy

    def _Q_x(self, x):
        """
        Derivatives of Q w.r.t x
        """
        gx = getattr(self, "_lng_%s"%x)
        qx = getattr(self, "_lnq_%s_"%x)

        return gx - qx

    @property
    def jacobian(self):
        """
        The jacobian of the current model and data, with respect to the parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`.

        See Murray, Power, Robotham Appendix for details. This is a 3-vector.
        """
        return np.sum(self._scaled_mass*np.array([self._Q_x("h"), self._Q_x("a"), self._Q_x("b")]),
                      axis=1)/self._mean_scaling

    @property
    def hessian(self):
        """
        The hessian of the current model and data, with respect to the parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`.

        See Murray, Power, Robotham Appendix for details. This is a 3x3 matrix.
        """
        hh = self._Q_x_y("h", "h")
        ha = self._Q_x_y("h", "a")
        hb = self._Q_x_y("h", "b")
        aa = self._Q_x_y("a", "a")
        ab = self._Q_x_y("a", "b")
        bb = self._Q_x_y("b", "b")
        hess = np.array([[hh, ha, hb],
                         [ha, aa, ab],
                         [hb, ab, bb]])

        return np.sum(self._scaled_mass*hess, axis=2)/self._mean_scaling

    @property
    def cov(self):
        """
        The covariance matrix of the current model and data, with respect to the parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`.

        Calculated numerically from the :method:`~hessian`. A 3x3 matrix.
        """
        return np.linalg.inv(-self.hessian)

    @property
    def corr(self):
        """
        The correlation matrix of the current model and data, with respect to the parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`.

        Calculated numerically from the :method:`~hessian`. A 3x3 matrix.
        """
        cov = self.cov
        s = np.sqrt(np.diag(cov))
        return cov/np.outer(s, s)


class PerObjLikeWeights(PerObjLike):
    """
    Compactified version of :class:`MRP_PO_Likelihood` useful for simulated haloes.

    Effectively, this is the same as :class:`MRP_PO_Likelihood`, but instead of
    passing a full array of halo masses, one can pass an array of unique masses,
    and a `weights` array which specifies the number of each mass in the sample.

    This should be useful for simulation-like data, which has many halos of the
    same mass.

    Parameters
    ----------
    weights : array_like
        Array of the same length as ``m``, giving the number of each mass in the
        distribution.

    Other Parameters
    ----------------
    args, kwargs :
        Other parameters are necessary, to be passed through to :class:`MRP_PO_Likelihood`.
    """

    def __init__(self, weights, *args, **kwargs):
        super(PerObjLikeWeights, self).__init__(*args, **kwargs)
        self.weights = weights

    @_cached
    def _scaled_mass(self):
        return self.weights*self.m**self.scale

    @_cached
    def _mean_scaling(self):
        return np.sum(self._scaled_mass)/np.sum(self.weights)


class CurveLike(PerObjLike):
    """
    A subclass of :class:`mrpy.core.MRP_PO_Likelihood` which adds the likelihood
    (and derivatives) of a model given data in the form of a curve.

    Parameters
    ----------
    logm : array_like
        Vector of log10 masses.

    logHs, alpha, beta : array_like
        The shape parameters of the MRP.

    scale : float, optional
        Mass scale with which to weight the individual masses. This can
        be useful to gain sensitivity to the high-mass haloes.

    norm : float or string
        Gives the normalisation of the MRP, *A*. If set to a *float*, it
        is directly the normalisation. If set to ``"pdf"``, it will automatically
        render the MRP as a statistical distribution. If set to ``"rhoc"``, it will
        yield the correct total mass density across all masses, down to ``m=0``.

    log_mmin : array_like, optional
        Log-10 truncation mass of the MRP. By default is set to the minimum mass
        in ``logm``.

    Om0 : float, optional
        Matter density of the Universe. Only required if `norm` is set to ``Arhoc``.

    rhoc : float, optional
        Crtical density of the Universe. Only required if `norm` is set to ``Arhoc``.

    dndm : array_like, optional
        Array of the same length as `m`, giving the value of the differential
        mass function. If `mw_data` is given, this is unnecessary to pass.

    mw_data : array_like, optional
        Array of the same length as `m`, giving the mass function weighted by
        `m**scale`. If passed, this takes priority over `dndm`. Either `dndm`
        or `mw_data` must be passed.

    mw_integ : float, optional
        The integral of `mw_data` over `m`. If not passed, will be calculated
        in the class. Fits are more computationally efficient if this is passed.

    sig_data : array_like, optional
        The uncertainty of the data (standard deviation). This is used in the likelihood
        to weight different mass scales. If scalar, all mass scales are weighted evenly.

    sig_integ, sig_rhomean,: float, optional
        These parameters control how much influence the integral of the data, and the
        total mean density of the universe, have on the likelihood. The default values
        of `inf` mean they are completely ignored. If they are 0, they become absolute
        constraints, so that either the data integral or total mass density of the universe
        is perfectly matched (setting the normalisation). In between, they act as
        uncertainties on those values.
    """

    def __init__(self, logm, logHs, alpha, beta,
                 dndm=None, mw_data=None, mw_integ=None,
                 sig_data=1, sig_integ=np.inf, sig_rhomean=np.inf,
                 scale=0, norm="pdf", log_mmin=None, Om0=0.3, rhoc=2.7755e11):

        super(CurveLike, self).__init__(logm, logHs, alpha, beta, scale, norm, log_mmin, Om0, rhoc)

        self.sig_rhomean = sig_rhomean
        self.sig_integ = sig_integ

        if dndm is None and mw_data is None:
            raise ValueError("At least one of dndm or mw_data must be specified")

        if mw_data is not None:
            self.mw_data = mw_data
        else:
            self.mw_data = dndm*self.m**self.scale

        self.Om0 = Om0
        self.rhoc = rhoc
        self.sig_data = sig_data
        self._mw_integ = mw_integ

        # Now correctly set alpha and A if necessary
        if sig_integ == 0 and sig_rhomean == 0:
            raise NotImplementedError()
            # A, self.alpha = get_alpha_and_A(self.logHs, self.beta, self.log_mmin,
            #                                 self.mw_integ, Om0, self.scale, rhoc)
            # self.lnA = np.log(A)
            # self._alpha_s = self.alpha + self.scale

        elif sig_integ == 0:
            self._norm = np.log(self.mw_integ) - np.log(self._q_) - self.scale*self.logHs*ln10
        elif sig_rhomean == 0:
            self._norm = np.log(core.A_rhoc(self.logHs, self.alpha, self.beta, Om0, rhoc))

    @_cached
    def _zk(self):
        """
        The z-factor for the total integral, i.e.: (alpha+2)/beta.
        """
        return (self.alpha + 2)/self.beta

    @_cached
    def mw_integ(self):
        if self._mw_integ is not None:
            return self._mw_integ
        else:
            return intg.simps(self.mw_data, self.m)

    # ===========================================================================
    # Basic likelihood
    # ===========================================================================
    @_cached
    def _fdata(self):
        return self._lng + self.scale*self.logHs*ln10 - np.log(self.mw_data)

    @_cached
    def _delta_data(self):
        """
        Distance between log theory and log data
        """
        return self.lnA + self._fdata

    @_cached
    def _fint(self):
        return -np.log(self.mw_integ/self._q_) + self.scale*self.logHs*ln10

    @_cached
    def _delta_integ(self):
        return self.lnA + self._fint

    @_cached
    def _frho(self):
        return - np.log(core.A_rhoc(self.logHs, self.alpha, self.beta, self.Om0, self.rhoc))

    @_cached
    def _delta_rhomean(self):
        return self.lnA + self._frho

    @_cached
    def _lnLi(self):
        """
        Logarithmic likelihood of the samples
        """
        base = self._delta_data**2/(2*self.sig_data**2)
        erri = 0
        errm = 0
        if (self.sig_rhomean == 0 or np.isinf(self.sig_rhomean)) and (self.sig_integ == 0 or np.isinf(self.sig_integ)):
            return -base
        else:
            if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
                errm = self._delta_rhomean**2/(2*self.sig_rhomean**2)
            if self.sig_integ != 0 and not np.isinf(self.sig_integ):
                erri = self._delta_integ**2/(2*self.sig_integ**2)
        return -(base + errm/len(self.m) + erri/len(self.m))

    # ===========================================================================
    # Simple lnk(theta) derivatives
    # ===========================================================================
    @_cached
    def _lnk(self):
        return np.log(self._k)

    @_cached
    def _lnk_h(self):
        return 2*ln10

    @_cached
    def _lnk_a(self):
        return sp.polygamma(0, self._zk)/self.beta

    @_cached
    def _lnk_b(self):
        return -self._lnk_a*self._zk

    @_cached
    def _lnk_h_h(self):
        return 0

    @_cached
    def _lnk_h_a(self):
        return 0

    @_cached
    def _lnk_h_b(self):
        return 0

    @_cached
    def _lnk_a_a(self):
        return sp.polygamma(1, self._zk)/self.beta**2

    @_cached
    def _lnk_a_b(self):
        return -(self._lnk_a_a*self._zk + self._lnk_a/self.beta)

    @_cached
    def _lnk_b_b(self):
        return self._lnk_a_a*self._zk**2 + 2*self._zk*self._lnk_a/self.beta

    @_cached
    def _lnk_jac(self):
        return np.array([getattr(self, "_lnk_%s"%x) for x in "hab"])

    @_cached
    def _lnk_hess(self):
        return np.array([getattr(self, "_lnk_%s_%s"%(x, y)) for x, y in ('hh', 'ha', 'hb',
                                                                         'ha', 'aa', 'ab',
                                                                         'hb', 'ab', 'bb')]).reshape((3, 3))

    # ===========================================================================
    # Jacobian etc.
    # ===========================================================================
    @_cached
    def _fdata_jac(self):
        return np.array([self._lng_h + self.scale*ln10, self._lng_a, self._lng_b])

    @_cached
    def _delta_data_jac(self):
        return np.vstack((self._fdata_jac, np.ones_like(self.m)))

    @_cached
    def _fint_jac(self):
        return np.array([self._lnq_h_ + self.scale*ln10, self._lnq_a_, self._lnq_b_])

    @_cached
    def _delta_integ_jac(self):
        return np.concatenate((self._fint_jac, [1]))

    @_cached
    def _frho_jac(self):
        return self._lnk_jac

    @_cached
    def _delta_rhomean_jac(self):
        return np.concatenate((self._frho_jac,[1]))

    @property
    def jacobian(self):
        """
        The Jacobian of the likelihood with respect to the MRP parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`, `[lnA]`.

        See Murray, Power, Robotham Appendix for details.

        .. note:: If the problem is constrained, this will be a length-3 vector, otherwise,
                  a length-4 vector including the normalisation.
        """
        data_term = np.sum(self._delta_data*self._delta_data_jac/self.sig_data**2, axis=-1)

        errm = np.zeros(4)
        erri = np.zeros(4)
        if self.sig_integ != 0 and not np.isinf(self.sig_integ):
            erri = self._delta_integ*self._delta_integ_jac/self.sig_integ**2
        if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
            errm = self._delta_rhomean*self._delta_rhomean_jac/self.sig_rhomean**2

        if self.sig_rhomean == 0 and self.sig_integ == 0:
            raise NotImplementedError("will have to be numerical eventually...")

        elif self.sig_rhomean == 0:
            J_u = -(data_term[:-1] + erri[:-1])
            return J_u + self._delta_rhomean_jac[:-1]*self._simple_sum
        elif self.sig_integ == 0:
            J_u = -(data_term[:-1] + errm[:-1])
            return J_u + self._delta_integ_jac[:-1]*self._simple_sum
        else:
            return -(data_term + erri + errm)

    @_cached
    def _simple_sum(self):
        data = np.sum(self._delta_data/self.sig_data**2)
        integ = 0
        rhom = 0
        if self.sig_integ != 0:
            integ = self._delta_integ/self.sig_integ**2
        if self.sig_rhomean != 0:
            rhom = self._delta_rhomean/self.sig_rhomean**2
        return data + integ + rhom

    # @cached
    # def _lng_xy(self):
    #     return np.array(
    #         [[getattr(self, "_lng_%s_%s"%(x if i < j else y, x if i > j else y)) for i, x in enumerate("hab")] for j, y
    #          in enumerate("hab")])

    @_cached
    def _lngx_lngy(self):
        return np.array([[getattr(self, "_lng_%s"%x)*getattr(self, "_lng_%s"%y) for x in "hab"] for y in "hab"])

    # @cached
    # def _lnq_xy(self):
    #     return np.array(
    #         [[getattr(self, "_lnq_%s_%s_"%(x if i < j else y, x if i > j else y)) for i, x in enumerate("hab")] for j, y
    #          in enumerate("hab")])

    @_cached
    def _lnqx_lnqy_(self):
        return np.array([[getattr(self, "_lnq_%s_"%x)*getattr(self, "_lnq_%s_"%y) for x in "hab"] for y in "hab"])

    # @cached
    # def _lnk_xy(self):
    #     return np.array(
    #         [[getattr(self, "_lnk_%s_%s"%(x if i < j else y, x if i > j else y)) for i, x in enumerate("hab")] for j, y
    #          in enumerate("hab")])

    @_cached
    def _lnkx_lnky(self):
        return np.array([[getattr(self, "_lnk_%s"%x)*getattr(self, "_lnk_%s"%y) for x in "hab"] for y in "hab"])

    @_cached
    def _lnkx_lngy(self):
        return np.array([[getattr(self, "_lnk_%s"%x)*getattr(self, "_lng_%s"%y) for x in "hab"] for y in "hab"])

    @_cached
    def _lnqx_lngy(self):
        return np.array([[getattr(self, "_lnq_%s_"%x)*getattr(self, "_lng_%s"%y) for x in "hab"] for y in "hab"])

    @_cached
    def _delta_data_jac_sq(self):
        dj = np.atleast_2d(self._delta_data_jac)
        return np.sum(np.einsum("ij,kj->ijk", dj, dj)/self.sig_data**2, axis=1)

    @_cached
    def _delta_integ_jac_sq(self):
        return np.outer(self._delta_integ_jac, self._delta_integ_jac)

    @_cached
    def _frho_jac_sq(self):
        return np.outer(self._frho_jac,self._frho_jac)

    @_cached
    def _delta_rhomean_jac_sq(self):
        return np.outer(self._delta_rhomean_jac, self._delta_rhomean_jac)

    @_cached
    def _fint_jac_sq(self):
        return np.outer(self._fint_jac,self._fint_jac)

    @_cached
    def _delta_data_hess(self):
        out = np.zeros((4, 4, len(self.m)))
        out[:-1, :-1, :] = self._lng_hess
        return out

    @_cached
    def _delta_integ_hess(self):
        out = np.zeros((4, 4))
        out[:-1, :-1] = self._lnq_hess_
        return out

    @_cached
    def _delta_rhomean_hess(self):
        out = np.zeros((4, 4))
        out[:-1, :-1] = self._lnk_hess
        return out

    @_cached
    def _gradsum(self):
        data = np.sum(self._fdata_jac/self.sig_data**2,axis=-1)
        integ = np.zeros(3)
        rhom = np.zeros(3)
        if self.sig_integ != 0:
            integ = self._fint_jac/self.sig_integ**2
        if self.sig_rhomean != 0:
            rhom = self._frho_jac/self.sig_rhomean**2
        return data + integ + rhom

    @property
    def hessian(self):
        """
        The hessian of the likelihood with respect to the MRP parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`, `[lnA]`.

        See Murray, Power, Robotham Appendix for details.

        .. note:: If the problem is constrained, this will be a 3x3 matrix, otherwise,
                  a 4x4 matrix including the normalisation.
        """
        # The basic layout here is that there are three possible terms (in the unconstrained case).
        # Each of these has the form [(del.delta)^2 + delta*del^2.delta]/sigma^2.
        # The data term has the extra complication that it needs to be summed over all bins.
        # However, when things are constrained it gets a bit more messy.

        ## Data term first
        errd = (self._delta_data_jac_sq + np.sum(self._delta_data_hess*self._delta_data, axis=-1))/self.sig_data**2

        ## Integral term
        erri = 0
        if self.sig_integ != 0 and not np.isinf(self.sig_integ):
            erri = (self._delta_integ_jac_sq + self._delta_integ_hess*self._delta_integ)/self.sig_integ**2

        ## Rhomean term
        errm = 0
        if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
            errm = (self._delta_rhomean_jac_sq + self._delta_rhomean_hess*self._delta_rhomean)/self.sig_rhomean**2

        if self.sig_rhomean == 0 and self.sig_integ == 0:
            raise NotImplementedError("will have to be numerical eventually...")
        elif self.sig_rhomean == 0:
            # TODO: Hessians for 3-vector cases.
            raise NotImplementedError()

            ## the following seems right, but doesn't give correct results in tests. More work needed...

            # data term first
            grad3_delta = np.sum((self._fdata_jac.T - self._frho_jac)/self.sig_data**2,axis=0)
            t1 = np.outer(grad3_delta,grad3_delta)
            t2 = np.sum(self._delta_data * (self._lng_hess.T - self._lnk_hess).T/self.sig_data**2,axis=-1)
            errd = t1+t2

            erri = 0
            ## Integ term if necessary
            if not np.isinf(self.sig_integ):
                grad3_delta = (self._fint_jac - self._frho_jac)/self.sig_int**2
                t1 = np.outer(grad3_delta,grad3_delta)
                t2 = self._delta_integ * (self._lnq_hess - self._lnk_hess)/self.sig_integ**2
                erri = t1+t2

            return -(errd+erri)

        elif self.sig_integ == 0:
            raise NotImplementedError()
        else:
            return -(errd + errm + erri)
