"""
Provides classes which extend the basic :class:`mrpy.core.MRP` class.

Adds methods for calculating the likelihood and its derivatives in special cases
of interest. Specifically, the two main cases of interest are fitting the MRP
to a sample of data (``SampleLike``) or fitting to a binned (or theoretical) curve.

At this point, the classes here only support the simplest possible cases, in which
the effective volume is constant as a function of mass, down to some threshold truncation mass.
Furthermore, only data without measurement error is supported at this point.

At this time, we don't directly support fitting MRP extensions, such as a double-MRP.
"""

import mrpy.base.special as sp
from mrpy.base import core
import numpy as np
import scipy.integrate as intg
from cached_property import cached_property as _cached
from mrpy.base import stats
from mrpy import MRP
from scipy.special import gamma

ln10 = np.log(10)


class SampleLike(core.MRP):
    """
    A subclass of :class:`mrpy.core.MRP` which adds the likelihood (and derivatives)
    of a model given a sample of masses.

    The likelihoods in this class are true under a number of simplifications. Firstly, the
    effective volume is constant down to a threshold minimum mass. Secondly, the masses
    have no measurement uncertainty. While these simplifications are rather
    restrictive, nevertheless the quantities here are suprisingly useful, for instance
    with estimating parameters for simulations, or as an ingredient in a more
    sophisticated analysis.

    Parameters
    ----------
    logm : array_like
        Vector of log10 masses.

    logHs, alpha, beta, lnA : array_like
        The parameters of the MRP.

    log_mmin : array_like, optional
        Log-10 truncation mass of the MRP. By default is set to the minimum mass
        in ``logm``.

    rhom : float
        Mass density of the universe. Only used if the normalisation is set to ``Arhom``.
    """

    # Internally, some of the properties are defined twice -- once for the
    # truncation mass, and once for all masses in logm. For example, ``x`` is
    # :math:`(m/H_s)^{\beta}, but for various purposes, ``m`` could be the full array
    # or just the truncation mass. Throughout, quantities that are defined as the
    # truncation mass have an extra trailing underscore in their name.

    def __init__(self, logm, logHs, alpha, beta, lnA, log_mmin=None, rhom =0.3 * 2.7755e11):
        super(SampleLike, self).__init__(logm, logHs, alpha, beta, lnA, log_mmin, rhom=rhom)

    def _getjac(self, var):
        X, Y = np.meshgrid(list(range(4)), list(range(4)))
        out = np.array(
            [getattr(self, "_%s_%s_%s%s"%(var.replace("_", ""), "habA"[min(x, y)], "habA"[max(x, y)],"_" if var.endswith("_") else "")) for x, y in
             zip(X.flatten(), Y.flatten())])

        if var.endswith("_"):
            return out.reshape((4, 4))
        else:
            return out.reshape((4, 4, len(self.m)))

    # ===========================================================================
    # Basic unit quantities
    # ===========================================================================
    @_cached
    def _z(self):
        """
        The term z = (1+alpha)/beta.
        """
        return (self.alpha + 1)/self.beta

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
        return self.A*self.Hs*sp.G1(self._z, self._x)

    @_cached
    def _G1_(self):
        return self.A*self.Hs*sp.G1(self._z, self._x_)

    @_cached
    def _G2(self):
        return self.A*self.Hs*sp.G2(self._z, self._x)

    @_cached
    def _G2_(self):
        return self.A*self.Hs*sp.G2(self._z, self._x_)

    @_cached
    def _Gbar(self):
        return self._G1*np.log(self._x) + 2*self._G2

    @_cached
    def _Gbar_(self):
        return self._G1_*np.log(self._x_) + 2*self._G2_

    @_cached
    def _phi(self):
        return self.m*self._g  # self._y*self._g/self._gammainc_zx

    @_cached
    def _phi_(self):
        return self.mmin*self._g_  # self._y_*self._g_/self._gammainc_zx_

    # ===========================================================================
    # Mass scaling utilities
    # ===========================================================================
    @_cached
    def _scaled_mass(self):
        return 1

    # ===========================================================================
    # Basic MRP quantities, renamed for compactness
    # ===========================================================================
    @_cached
    def _g(self):
        """
        The MRP
        """
        return self.dndm()

    @_cached
    def _g_(self):
        """
        The MRP at truncation mass
        """
        return core.dndm(self.mmin, self.logHs, self.alpha, self.beta, self.mmin, norm=self.A)

    @_cached
    def _lng(self):
        """
        Better log of g than log(g) (all masses)
        """
        return self.dndm(True)

    @_cached
    def _lng_(self):
        """
        Better log of g than log(g) (truncation mass)
        """
        return core.dndm(self.mmin, self.logHs, self.alpha, self.beta, self.mmin, norm=self.A, log=True)

    @_cached
    def _q(self):
        """
        The normalisation of the MRP (ie. integral of g) (all masses)
        """
        return self.A * stats.TGGD(scale=self.Hs, a=self.alpha, b=self.beta, xmin=self.m)._pdf_norm()

    @_cached
    def _q_(self):
        """
        The normalisation of the MRP (ie. integral of g) (truncation mass)
        """
        return self.A * stats.TGGD(scale=self.Hs, a=self.alpha, b=self.beta, xmin=self.mmin)._pdf_norm()

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


    @property
    def lnL(self):
        """
        Total log-likelihood with current model for masses m [uniform prior]
        """
        return np.sum(self._scaled_mass * self._lng) - self._q_

    # ===========================================================================
    # Simple Derivatives
    # ===========================================================================
    # ----------- g'() --------------------------------------
    @_cached
    def _lng_A(self):
        return np.ones_like(self.m)

    @_cached
    def _lng_A_(self):
        return 1

    @_cached
    def _lng_h(self):
        return (self.beta*self._x - self.alpha)*ln10

    @_cached
    def _lng_h_(self):
        return (self.beta*self._x_ - self.alpha)*ln10

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
    def _lng_A_A(self):
        return np.zeros_like(self.m)

    @_cached
    def _lng_A_A_(self):
        return 0

    @_cached
    def _lng_a_A(self):
        return np.zeros_like(self.m)

    @_cached
    def _lng_a_A_(self):
        return 0

    @_cached
    def _lng_b_A(self):
        return np.zeros_like(self.m)

    @_cached
    def _lng_b_A_(self):
        return 0

    @_cached
    def _lng_h_A(self):
        return np.zeros_like(self.m)

    @_cached
    def _lng_h_A_(self):
        return 0

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
        return np.array([getattr(self, "_lng_%s"%x) for x in 'habA'])

    @_cached
    def _lng_hess(self):
        return self._getjac("_lng")

    @_cached
    def _lng_jac_(self):
        return np.array([getattr(self, "_lng_%s_"%x) for x in 'habA'])

    @_cached
    def _lng_hess_(self):
        return self._getjac("_lng_")

    # --------- \tilde{q}'() --------------------------------------
    @_cached
    def _q_A(self):
        return self._q

    @_cached
    def _q_A_(self):
        return self._q_


    @_cached
    def _q_h(self):
        return ln10*(self._q + self._phi)

    @_cached
    def _q_h_(self):
        return ln10*(self._q_ + self._phi_)

    @_cached
    def _q_a(self):
        return (self._q*np.log(self._x) + self._G1)/self.beta

    @_cached
    def _q_a_(self):
        return (self._q_*np.log(self._x_) + self._G1_)/self.beta

    @_cached
    def _q_b(self):
        return -self._z*self._q_a - self._phi*np.log(self._y)/self.beta

    @_cached
    def _q_b_(self):
        return -self._z*self._q_a_ - self._phi_*np.log(self._y_)/self.beta

    @_cached
    def _q_A_A(self):
        return self._q

    @_cached
    def _q_h_A(self):
        return self._q_h

    @_cached
    def _q_a_A(self):
        return self._q_a

    @_cached
    def _q_b_A(self):
        return self._q_b

    @_cached
    def _q_A_A_(self):
        return self._q_

    @_cached
    def _q_h_A_(self):
        return self._q_h_

    @_cached
    def _q_a_A_(self):
        return self._q_a_

    @_cached
    def _q_b_A_(self):
        return self._q_b_

    @_cached
    def _q_h_h(self):
        return ln10*(self._q_h + self._phi*self._lng_h)

    @_cached
    def _q_h_h_(self):
        return ln10*(self._q_h_ + self._phi_*self._lng_h_)

    @_cached
    def _q_a_a(self):
        return (1.0/self.beta)*(self._q_a*np.log(self._x) + self._Gbar/self.beta)

    @_cached
    def _q_a_a_(self):
        return (1.0/self.beta)*(self._q_a_*np.log(self._x_) + self._Gbar_/self.beta)

    @_cached
    def _q_a_b(self):
        b = self.beta
        lnx = np.log(self._x)
        return (1.0/self.beta)*(-self._q_a + self._q_b*lnx - (self._z*self._Gbar/b))

    @_cached
    def _q_a_b_(self):
        b = self.beta
        lnx = np.log(self._x_)
        return (1.0/self.beta)*(-self._q_a_ + self._q_b_*lnx - (self._z*self._Gbar_/b))

    @_cached
    def _q_b_b(self):
        b = self.beta
        return (self._phi*np.log(self._y)/b)*(1./b - self._lng_b) - self._z*(self._q_a_b - self._q_a/b)

    @_cached
    def _q_b_b_(self):
        b = self.beta
        return (self._phi_*np.log(self._y_)/b)*(1./b - self._lng_b_) - self._z*(self._q_a_b_ - self._q_a_/b)

    @_cached
    def _q_h_a(self):
        return ln10*(self._q_a + self._phi*self._lng_a)

    @_cached
    def _q_h_a_(self):
        return ln10*(self._q_a_ + self._phi_*self._lng_a_)

    @_cached
    def _q_h_b(self):
        return ln10*(self._q_b + self._phi*self._lng_b)

    @_cached
    def _q_h_b_(self):
        return ln10*(self._q_b_ + self._phi_*self._lng_b_)

    @_cached
    def _q_jac(self):
        return np.array([getattr(self, "_q_%s"%x) for x in 'habA'])

    @_cached
    def _q_jac_(self):
        return np.array([getattr(self, "_q_%s_"%x) for x in 'habA'])

    @_cached
    def _q_hess(self):
        return self._getjac("_q")

    @_cached
    def _q_hess_(self):
        return self._getjac("_q_")

    # ------ Matrices ------------------
    # def _Q_x_y(self, x, y):
    #     """
    #     Double derivatives of Q w.r.t x then y
    #     """
    #     if y == "h" or (y == "a" and x == "b"):
    #         x, y = y, x
    #
    #     lngxy = getattr(self, "_lng_%s_%s"%(x, y))
    #     qxy = getattr(self, "_q_%s_%s_"%(x, y))
    #
    #     return lngxy - qxy
    #
    # def _Q_x(self, x):
    #     """
    #     Derivatives of Q w.r.t x
    #     """
    #     gx = getattr(self, "_lng_%s"%x)
    #     qx = getattr(self, "_q_%s_"%x)
    #
    #     return gx - qx

    @property
    def jacobian(self):
        """
        The jacobian of the current model and data, with respect to the parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`.

        See Murray, Power, Robotham Appendix for details. This is a 3-vector.
        """
        return np.sum(self._scaled_mass * self._lng_jac,axis=1) - self._q_jac_
#        return np.sum(self._scaled_mass * np.array([self._Q_x(x) for x in "habA"]), axis=1)

    @property
    def hessian(self):
        """
        The hessian of the current model and data, with respect to the parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`.

        See Murray, Power, Robotham Appendix for details. This is a 3x3 matrix.
        """
        return np.sum(self._scaled_mass * self._lng_hess, axis=2) - self._q_hess_#/self._mean_scaling

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


class SampleLikeWeights(SampleLike):
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
        super(SampleLikeWeights, self).__init__(*args, **kwargs)
        self.weights = weights

    @_cached
    def _scaled_mass(self):
        return self.weights


def expected_likelihood(theta, data_m, data_mf, kappa=None, V0=1, mmin=None):
    h,a,b,lnA = theta

    A = np.exp(lnA)

    if kappa is None:
        kappa = 0
        mmin = data_m[0]

    if a+1+kappa < 0 and mmin is None:
        raise Exception("Non convergent integral")

    mrp = MRP(data_m, h, a+kappa, b, norm=lnA + np.log(V0) + np.log(10)*kappa*h, log_mmin=mmin)#.dndm() * 10 ** x * np.log(10)


    # If slope is non-convergent, trust we have an mmin, else use convergent integral
    if a + kappa + 1 < 0:
        q = mrp.nbar
    else:
        q = 10 ** (h * (kappa + 1)) * V0 * A * gamma((a + kappa + 1) / b)

    integral = intg.simps(V0 * 10**(kappa*data_m) * data_mf * (mrp.dndlog10m(True)), data_m)

    return -q + integral


class CurveLike(SampleLike):
    """
    A subclass of :class:`mrpy.core.MRP_PO_Likelihood` which adds the likelihood
    (and derivatives) of a model given data in the form of a curve.

    See Murray, Robotham, Power (2017), Appendix C.1 for a description.

    Parameters
    ----------
    logm : array_like
        Vector of log10 masses.

    data_dndm : array_like
        Array of the same length as `logm`, giving the value of the differential
        mass function.

    hs, alpha, beta, lnA : float
        The parameters of the MRP.

    sig_data : array_like, optional
        The uncertainty of the data (standard deviation). This is used in the likelihood
        to weight different mass scales. If scalar, all mass scales are weighted evenly.

    sig_rhomean,: float, optional
        This controls how much influence the total mean density of the universe has
        on the likelihood. The default value of `inf` means it is completely ignored.
        If it is 0, it becomes an absolute constraint, so that the total mass density
        of the universe is perfectly matched (setting the normalisation). In between,
        it acts as an uncertainty on this value.

    rhom : float, optional
        Mass density of the Universe. Only used if 'sig_rhomean` not infinite.

    """

    def __init__(self, logm, data_dndm, logHs, alpha, beta, lnA,
                 sig_data=1, sig_rhomean=np.inf,
                 rhom=0.3* 2.7755e11):

        super(CurveLike, self).__init__(logm, logHs, alpha, beta, lnA, None, rhom=rhom)

        self.sig_rhomean = sig_rhomean

        self.data_dndm = data_dndm

        self.rhom = rhom
        self.sig_data = sig_data


    @_cached
    def _zk(self):
        """
        The z-factor for the total integral, i.e.: (alpha+2)/beta.
        """
        return (self.alpha + 2)/self.beta

    # ===========================================================================
    # Basic likelihood
    # ===========================================================================
    @_cached
    def _fdata(self):
        return self._lng - np.log(self.data_dndm)

    @_cached
    def _delta_data(self):
        """
        Distance between log theory and log data
        """
        return self._fdata

    @_cached
    def _frho(self):
        return - np.log(core.A_rhom(self.logHs, self.alpha, self.beta, self.rhom))

    @_cached
    def _delta_rhomean(self):
        return self.lnA + self._frho

    @_cached
    def _lnLi(self):
        """
        Logarithmic likelihood of the samples
        """
        base = self._delta_data**2/(2*self.sig_data**2)

        errm = 0
        if (self.sig_rhomean == 0 or np.isinf(self.sig_rhomean)):
            return -base
        else:
            if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
                errm = self._delta_rhomean**2/(2*self.sig_rhomean**2)

        return -(base + errm/len(self.m))

    @_cached
    def lnL(self):
        return np.sum(self._lnLi)

    # ===========================================================================
    # Simple lnk(theta) derivatives
    # ===========================================================================
    @_cached
    def _lnk(self):
        return np.log(self._k)

    @_cached
    def _lnk_A_(self):
        return 0

    @_cached
    def _lnk_A_A_(self):
        return 0

    @_cached
    def _lnk_h_A_(self):
        return 0

    @_cached
    def _lnk_b_A_(self):
        return 0

    @_cached
    def _lnk_a_A_(self):
        return 0

    @_cached
    def _lnk_h_(self):
        return 2*ln10

    @_cached
    def _lnk_a_(self):
        return sp.polygamma(0, self._zk)/self.beta

    @_cached
    def _lnk_b_(self):
        return -self._lnk_a_*self._zk

    @_cached
    def _lnk_h_h_(self):
        return 0

    @_cached
    def _lnk_h_a_(self):
        return 0

    @_cached
    def _lnk_h_b_(self):
        return 0

    @_cached
    def _lnk_a_a_(self):
        return sp.polygamma(1, self._zk)/self.beta**2

    @_cached
    def _lnk_a_b_(self):
        return -(self._lnk_a_a_*self._zk + self._lnk_a_/self.beta)

    @_cached
    def _lnk_b_b_(self):
        return self._lnk_a_a_*self._zk**2 + 2*self._zk*self._lnk_a_/self.beta

    @_cached
    def _lnk_jac(self):
        return np.array([getattr(self, "_lnk_%s_"%x) for x in "hab"])

    @_cached
    def _lnk_hess(self):
        return self._getjac("_lnk_")

    # ===========================================================================
    # Jacobian etc.
    # ===========================================================================
    @_cached
    def _fdata_jac(self):
        return np.array([self._lng_h, self._lng_a, self._lng_b])

    @_cached
    def _delta_data_jac(self):
        return np.vstack((self._fdata_jac, np.ones_like(self.m)))

    @_cached
    def _frho_jac(self):
        return self._lnk_jac

    @_cached
    def _delta_rhomean_jac(self):
        return np.concatenate((self._frho_jac, [1]))

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
        if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
            errm = self._delta_rhomean*self._delta_rhomean_jac/self.sig_rhomean**2

        if self.sig_rhomean == 0:
            J_u = -(data_term[:-1] + erri[:-1])
            return J_u + self._delta_rhomean_jac[:-1]*self._simple_sum
        else:
            return -(data_term + errm)

    @_cached
    def _simple_sum(self):
        data = np.sum(self._delta_data/self.sig_data**2)
        integ = 0
        rhom = 0

        if self.sig_rhomean != 0:
            rhom = self._delta_rhomean/self.sig_rhomean**2
        return data + rhom

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
        return np.array([[getattr(self, "_q_%s_"%x)*getattr(self, "_q_%s_"%y) for x in "hab"] for y in "hab"])/self._q_

    # @cached
    # def _lnk_xy(self):
    #     return np.array(
    #         [[getattr(self, "_lnk_%s_%s"%(x if i < j else y, x if i > j else y)) for i, x in enumerate("hab")] for j, y
    #          in enumerate("hab")])

    @_cached
    def _lnkx_lnky(self):
        return np.array([[getattr(self, "_lnk_%s_"%x)*getattr(self, "_lnk_%s_"%y) for x in "hab"] for y in "hab"])

    @_cached
    def _lnkx_lngy(self):
        return np.array([[getattr(self, "_lnk_%s"%x)*getattr(self, "_lng_%s"%y) for x in "hab"] for y in "hab"])

    @_cached
    def _lnqx_lngy(self):
        return np.array([[getattr(self, "_q_%s_"%x)*getattr(self, "_lng_%s"%y) for x in "hab"] for y in "hab"])/self._q_

    @_cached
    def _delta_data_jac_sq(self):
        dj = np.atleast_2d(self._delta_data_jac)
        return np.sum(np.einsum("ij,kj->ijk", dj, dj)/self.sig_data**2, axis=1)

    @_cached
    def _frho_jac_sq(self):
        return np.outer(self._frho_jac, self._frho_jac)

    @_cached
    def _delta_rhomean_jac_sq(self):
        return np.outer(self._delta_rhomean_jac, self._delta_rhomean_jac)


    @_cached
    def _delta_data_hess(self):
        out = self._lng_hess
        return out

    @_cached
    def _delta_rhomean_hess(self):
        out = self._lnk_hess
        return out

    @_cached
    def _gradsum(self):
        data = np.sum(self._fdata_jac/self.sig_data**2, axis=-1)
        rhom = np.zeros(3)

        if self.sig_rhomean != 0:
            rhom = self._frho_jac/self.sig_rhomean**2

        return data + rhom

    @property
    def hessian(self):
        """
        The hessian of the likelihood with respect to the MRP parameters.

        Order of the parameters is `logHs`, `alpha`, `beta`, `[lnA]`.

        See Murray, Power, Robotham Appendix for details.

        .. note:: If the problem is constrained, this will be a 3x3 matrix, otherwise,
                  a 4x4 matrix including the normalisation.
        """
        # The basic layout here is that there are two possible terms (in the unconstrained case).
        # Each of these has the form [(del.delta)^2 + delta*del^2.delta]/sigma^2.
        # The data term has the extra complication that it needs to be summed over all bins.
        # However, when things are constrained it gets a bit more messy.

        ## Data term first
        errd = (self._delta_data_jac_sq + np.sum(self._delta_data_hess*self._delta_data, axis=-1))/self.sig_data**2


        ## Rhomean term
        errm = 0
        if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
            errm = (self._delta_rhomean_jac_sq + self._delta_rhomean_hess*self._delta_rhomean)/self.sig_rhomean**2

        if self.sig_rhomean == 0:
            # TODO: Hessians for 3-vector cases.
            raise NotImplementedError()

            ## the following seems right, but doesn't give correct results in tests. More work needed...

            # data term first
            grad3_delta = np.sum((self._fdata_jac.T - self._frho_jac)/self.sig_data**2, axis=0)
            t1 = np.outer(grad3_delta, grad3_delta)
            t2 = np.sum(self._delta_data*(self._lng_hess.T - self._lnk_hess).T/self.sig_data**2, axis=-1)
            errd = t1 + t2

            erri = 0

            return -(errd + erri)


        else:
            return -(errd + errm)
