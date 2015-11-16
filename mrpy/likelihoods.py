"""
Defines a couple of classes which implement the MRP, along with derived quantities
and things like Jacobians/Hessians with respect to parameters.
"""
from special import gamma, gammainc, polygamma, _mg, hyperReg, hyperReg_vec
from mpmath import meijerg

import core
import numpy as np
from scipy.integrate import simps
from cached_property import cached_property as cached
ln10 = np.log(10)

class MRP(object):
    """
    A class defining the basic MRP functions.

    Parameters
    ----------
    logm : array_like
        Vector of log10 masses.

    logHs, alpha, beta : float
        The shape parameters of the MRP.

    log_mmin : float
        Log-10 truncation mass of the MRP. By default is set to the minimum mass
        in ``logm``.
    """
    def __init__(self, logm, logHs, alpha, beta, log_mmin=None ):
        self.logm = logm
        self.log_mmin = log_mmin or logm.min()
        self.logHs = float(logHs)
        self.alpha = float(alpha)
        self.beta = float(beta)
#        self.scale = scale
#        self._alpha_s = self.alpha + self.scale

#        self._shape = [getattr(getattr(self, q), "__len__", None) for q in ["log_mmin", "logHs", "alpha", "beta", "scale"]]
#        while None in self._shape:
#            self._shape.remove(None)

    @property
    def m(self):
        """
        Un-logged masses
        """
        return 10**self.logm

    @property
    def mmin(self):
        """
        Un-logged truncation mass
        """
        return 10**self.log_mmin

    @property
    def Hs(self):
        """
        Unlogged scale mass.
        """
        return 10**self.logHs

    @cached
    def log_mass_mode(self):
        r"""
        The (log) mode of the MRP weighted by mass in log-space:
        :math:`H_s \sqrt{\beta}{z+1/\beta}`.
        """
        return core.log_mass_mode(self.logHs,self.alpha,self.beta)


    def mrp(self,mmax=np.inf,norm="pdf",log=False,**Arhoc_kw):
        """
        Return the MRP.

        It is normalised by the parameter ``norm``. A float is interpreted directly
        as ``A``, whereas, the string ``pdf`` is interpreted as integrating to 1
        between mmin and mmax. The string "rhoc" forces the integral to converge
        to the total mass density of the Universe, specified in ``Arhoc_kw``.

        Parameters
        ----------
        mmax : float, optional
            Log-10 of the upper-truncation mass. Default is infinite mass.

        norm : None, string or float

        log : logical
            Whether to return the natural log of the MRP (suitable for Bayesian
            likelihoods).

        \*\*Arhoc_kw :
            Arguments directly forwarded to the mean-density normalisation.
        """
        return core.mrp(self.m, self.logHs,self.alpha,self.beta,mmin=self.log_mmin,
                        mmax=mmax,norm=norm,log=log,**Arhoc_kw)

    def ngtm(self,mmax=np.inf,norm="cdf",log=False,**Arhoc_kw):
        """
        The CDF of the MRP, in reverse (i.e. CDF=1 at mmin).

        It is normalised by the parameter ``norm``. A float is interpreted directly
        as ``A``, whereas, the string ``pdf`` is interpreted as integrating to 1
        between mmin and mmax. The string "rhoc" forces the integral to converge
        to the total mass density of the Universe, specified in ``Arhoc_kw``.

        Parameters
        ----------
        mmax : float, optional
            Log-10 of the upper-truncation mass. Default is infinite mass.

        norm : None, string or float

        log : logical
            Whether to return the natural log of the MRP (suitable for Bayesian
            likelihoods).

        \*\*Arhoc_kw :
            Arguments directly forwarded to the mean-density normalisation.
        """
        return core.ngtm(self.m, self.logHs,self.alpha,self.beta,mmin=self.log_mmin,
                         mmax=mmax,norm=norm,log=log,**Arhoc_kw)

    @cached
    def _k(self):
        """
        The integral of the mass-weighted MRP over all masses.
        """
        return core.k(self.logHs,self.alpha,self.beta)

    def A_rhoc(self,Om0=0.3,rhoc=2.7755e11):
        """
        The normalisation of the MRP to retrieve a total mass density of Omega_CDM.
        """
        return Om0 * rhoc / self._k

    def nbar(self,norm="rhoc",**Arhoc_kw):
        """
        Total number density above mmin, normalised according to ``norm``.

        By default, gives the density which integrates to the total mass density
        of the Universe.
        """
        return core.ngtm(self.mmin, self.logHs,self.alpha,self.beta,mmin=self.log_mmin,
                         mmax=np.inf,norm=norm,log=False,**Arhoc_kw)


    def rho_gtm(self,mmax=np.inf,norm="pdf",log=False,**Arhoc_kw):
        """
        The mass-weighted integral of the MRP, in reverse (ie. from high to low mass).

        It is normalised by the parameter ``norm``. A float is interpreted directly
        as ``A``, whereas, the string ``pdf`` is interpreted as integrating to 1
        between mmin and mmax. The string "rhoc" forces the integral to converge
        to the total mass density of the Universe, specified in ``Arhoc_kw``.

        Parameters
        ----------
        mmax : float, optional
            Log-10 of the upper-truncation mass. Default is infinite mass.

        norm : None, string or float

        log : logical
            Whether to return the natural log of the MRP (suitable for Bayesian
            likelihoods).

        \*\*Arhoc_kw :
            Arguments directly forwarded to the mean-density normalisation.
        """
        return core.rho_gtm(self.m, self.logHs,self.alpha,self.beta,mmin=self.log_mmin,
                            mmax=mmax,norm=norm,log=log,**Arhoc_kw)

    def rhobar(self,norm="rhoc",**Arhoc_kw):
        """
        Total mass density above mmin, normalised according to ``norm``.

        By default, gives the density which integrates to the total mass density
        of the Universe.
        """
        return core.rho_gtm(self.mmin, self.logHs,self.alpha,self.beta,mmin=self.log_mmin,
                            mmax=mmax,norm=norm,log=log,**Arhoc_kw)

    #
    # @cached
    # def mrp_norm(self, m):
    #     """
    #     The MRP normalised to have the correct total mass density over all mass
    #     """
    #     return self.norm * self.g
    #
    def generate_masses(self, N):
        """
        Return a list of halo masses sampled from current parameters
        """
        mmax = self.Hs * (600.0) ** (1 / self.beta)

        m = 10 ** np.linspace(np.log10(self.mmin), np.log10(mmax), 2000)
        cdf = self.ngtm(m,norm="cdf")
        icdf = spline(cdf[::-1], np.log10(m[::-1]), k=2)

        # if N is None:
        #     N = int(self.nbar * self.V)

        x = np.random.random(N)
        y = 10 ** icdf(x)

        # Just to make sure everything's ok
        i = 1
        while len(y[y < 0]) > 0 or len(y[y > mmax]) > 0:
            i *= 2
            print "Increasing Resolution..."
            m = 10 ** np.linspace(np.log10(self.mmin), np.log10(mmax), 5000 * i)
            cdf = self.ngtm(m,norm="cdf")
            icdf = spline(cdf[::-1], np.log10(m[::-1]), k=2)
            y = 10 ** icdf(x)

        return y


class MRP_PO_Likelihood(MRP):
    """
    A subclass of MRP which adds likelihoods and Jacobians/Hessians for samples
    of individual masses.

    In this case, the fitting process can be generalised to be weighted by some
    power of the mass, which may improve fitted quality. We include this power,
    s, in this class. This scaling doesn't touch the standard MRP outputs, just
    the quantities involved in fitting introduced here.

    Internally, some of the properties are defined twice -- once for the
    truncation mass, and once for all masses in logm. For example, ``x`` is
    :math:`(m/H_s)^beta`, but for various purposes, ``m`` could be the full array
    or just the truncation mass. Throughout, quantities that are defined as the
    truncation mass have an extra trailing underscore in their name.

    Parameters
    ----------
    scale : float, optional
        Mass scale, s, with which to weight the individual masses.
    """
    def __init__(self,scale=0,*args,**kwargs):
        self.scale=scale
        super(MRP_PO_Likelihood,self).__init__(*args,**kwargs)
        self._alpha_s = self.alpha + scale

    #===========================================================================
    # Basic unit quantities
    #===========================================================================
    @cached
    def _z(self):
        """
        The term z = (1+alpha)/beta.
        """
        return (self._alpha_s + 1) / self.beta

    @cached
    def _y_(self):
        """
        The scaled truncation mass
        """
        return self.mmin / self.Hs

    @cached
    def _x_(self):
        """
        y^beta (truncation mass scaled)
        """
        return self._y_ ** self.beta

    @cached
    def _y(self):
        """
        The scaled masses
        """
        return self.m / self.Hs

    @cached
    def _x(self):
        """
        y^beta (all masses, not just truncation)
        """
        return self._y ** self.beta

    #===========================================================================
    # Define some specific special quantities
    #===========================================================================
    def _hyperReg(self, a, b, z):
        # Note the following ONLY works properly for the calls in this class
        if hasattr(a[0], "__len__"):
            x = copy(a)
            y = copy(b)
            out = np.zeros_like(a[0])
            z = np.atleast_1d(z)
            if len(z) == 1:
                z = np.repeat(z, len(a[0]))

            for i, aa, bb, zz in enumerate(zip(a[0], b[0], z)):
                x[0] = aa
                x[1] = aa
                y[0] = bb
                y[1] = bb
                out[i] = float(hyperReg(x, y, zz))
            return out
        elif hasattr(z, "__len__"):
            a_s = np.ndarray((1,), dtype=object)
            a_s[0] = a
            b_s = np.ndarray((1,), dtype=object)
            b_s[0] = b
            return hyperRegA(a_s, b_s, z).astype("float")

        else:
            return hyperReg(a, b, z)

    def _meijerg(self, a, b, z):
        # Note the following ONLY works properly for G1 and G2... not MeijerG in general.
        if hasattr(b[0][-1], "__len__"):
            x = copy(b)
            out = np.zeros_like(b[0][-1])
            z = np.atleast_1d(z)
            if len(z) == 1:
                z = np.repeat(z, len(b[0][-1]))
            for i, (bb, zz) in enumerate(zip(b[0][-1], z)):
                x[0][-1] = bb
                out[i] = float(meijerg(a, x, zz))
            return out
        elif hasattr(z, "__len__"):
            a_s = np.ndarray((1,), dtype=object)
            a_s[0] = a
            b_s = np.ndarray((1,), dtype=object)
            b_s[0] = b
            return _mg(a_s, b_s, z).astype("float")

        else:
            return float(meijerg(a, b, z))


    #===========================================================================
    # Cached special functions
    #===========================================================================
    @cached
    def _gammaz(self):
        """
        The gamma function at z=(1+a)/b. Stored for use elsewhere.
        """
        return gamma(self._z)

    @cached
    def _gammainc_zx(self):
        """
        The incomplete gamma function, Gamma(z,x), where z,x are as specified in
        this class.
        """
        return gammainc(self._z, self._x)

    @cached
    def _gammainc_zx_(self):
        """
        The incomplete gamma function, Gamma(z,x), where z,x are as specified in
        this class.
        """
        return gammainc(self._z, self._x_)

    @cached
    def _G1(self):
        return self._meijerg([[], [1, 1]], [[0, 0, self._z], []], self._x_)/self._gammainc_zx_

    @cached
    def _G2(self):
        return self._meijerg([[], [1, 1, 1]], [[0, 0, 0, self._z], []], self._x_)/self._gammainc_zx_

    @cached
    def _Gbar(self):
        return self._G1*np.log(self._x_)+2*self._G2

    @cached
    def _phi(self):
        return self._y_*self._g_/self._gammainc_zx_

    #===========================================================================
    # Mass scaling utilities
    #===========================================================================
    @cached
    def scaled_mass(self):
        return self.m ** self.scale

    @cached
    def mean_scaling(self):
        return np.mean(self.scaled_mass)


    #===========================================================================
    # Basic MRP quantities, renamed for compactness
    #===========================================================================
    @cached
    def _g(self):
        """
        The shape of the MRP, completely unnormalised (ie. A=1) (all masses)
        """
        return core.mrp_shape(self.m,self.logHs,self._alpha_s,self.beta)

    @cached
    def _g_(self):
        """
        The shape of the MRP, completely unnormalised (ie. A=1) (truncation mass)
        """
        return core.mrp_shape(self.mmin,self.logHs,self._alpha_s,self.beta)

    @cached
    def _lng(self):
        """
        Better log of g than log(g) (all masses)
        """
        return core.ln_mrp_shape(self.m,self.logHs,self._alpha_s,self.beta)

    @cached
    def _lng_(self):
        """
        Better log of g than log(g) (truncation mass)
        """
        return core.ln_mrp_shape(self.mmin,self.logHs,self._alpha_s,self.beta)

    @cached
    def _q(self):
        """
        The normalisation of the MRP (ie. integral of g) (all masses)
        """
        return gammainc(self._z, self._x) * self.Hs

    @cached
    def _q_(self):
        """
        The normalisation of the MRP (ie. integral of g) (truncation masses)
        """
        return gammainc(self._z, self._x_) * self.Hs


    #===========================================================================
    # Basic likelihood
    #===========================================================================
    @cached
    def _lnLi(self):
        """
        Logarithmic likelihood of the particles, given its mass and a model (uniform prior)
        """
        return self.scaled_mass * (self._lng - np.log(self._q_)) / self.mean_scaling

    @cached
    def lnL(self):
        """
        Total log-likelihood with current model for masses m [uniform prior]

        If m is None, a sampling from the current model is used to get m
        """
        return np.sum(self._lnLi)


    #===========================================================================
    # Simple Derivatives
    #===========================================================================
    #----------- g'() --------------------------------------
    @cached
    def _lng_h(self):
        return (self.beta * self._x-self._alpha_s) * ln10

    @cached
    def _lng_h_(self):
        return (self.beta * self._x_-self._alpha_s) * ln10

    @cached
    def _lng_a(self):
        return np.log(self._y)

    @cached
    def _lng_a_(self):
        return np.log(self._y_)

    @cached
    def _lng_b(self):
        return (1  - self._x * np.log(self._x))/ self.beta

    @cached
    def _lng_b_(self):
        return (1  - self._x_ * np.log(self._x_))/ self.beta

    @cached
    def _lng_h_h(self):
        return -ln10**2*self.beta**2*self._x

    @cached
    def _lng_a_a(self):
        return np.zeros(len(self.m))

    @cached
    def _lng_a_b(self):
        return np.zeros(len(self.m))

    @cached
    def _lng_b_b(self):
        return -1 / self.beta ** 2 - self._x * np.log(self._y) ** 2

    @cached
    def _lng_h_a(self):
        return -np.ones(len(self.m)) * ln10

    @cached
    def _lng_h_b(self):
        return self._x * (1 + np.log(self._x)) * ln10


    #--------- \tilde{q}'() --------------------------------------
    @cached
    def _lnq_h(self):
        return ln10*(1+self._phi)

    @cached
    def _lnq_a(self):
        return (np.log(self._x_)+self._G1)/self.beta

    @cached
    def _lnq_b(self):
        return -self._z*self._lnq_a - self._phi*np.log(self._y_)/self.beta

    @cached
    def _q_h_h(self):
        return ln10*(self._lnq_h+self._phi*self._lng_h_)

    @cached
    def _q_a_a(self):
        return (1.0 / self.beta) * (self._lnq_a*np.log(self._x_) + self._Gbar/self.beta)

    @cached
    def _q_a_b(self):
        b = self.beta
        lnx = np.log(self._x_)
        return (1.0 / self.beta) * (-self._lnq_a + self._lnq_b * lnx - (self._z * self._Gbar / b))

    @cached
    def _q_b_b(self):
        b = self.beta
        return (self._phi*np.log(self._y_) / b) * (1./b-self._lng_b_)- self._z*(self._q_a_b-self._lnq_a/b)

    @cached
    def _q_h_a(self):
        return ln10*(self._lnq_a + self._phi*self._lng_a_)

    @cached
    def _q_h_b(self):
        return ln10*(self._lnq_b + self._phi*self._lng_b_)


    #------ Matrices ------------------
    def _Q_x_y(self, x, y):
        """
        Double derivatives of Q w.r.t x then y
        """
        if y == "h" or (y == "a" and x == "b"):
            x, y = y, x

        lngxy = getattr(self, "_lng_%s_%s" % (x, y))
        lnqx = getattr(self, "_lnq_%s" % x)
        lnqy = getattr(self, "_lnq_%s" % y)
        qxy = getattr(self, "_q_%s_%s" % (x, y))

        lnqxy = qxy - lnqx*lnqy

        return  lngxy - lnqxy

    def _Q_x(self, x):
        """
        Derivatives of Q w.r.t x
        """
        gx = getattr(self, "_lng_%s" % x)
        qx = getattr(self, "_lnq_%s" % x)

        return gx - qx

    @cached
    def jacobian(self):
        return np.sum(self.scaled_mass * np.array([self._Q_x("h"), self._Q_x("a"), self._Q_x("b")]), axis=1) / self.mean_scaling

    @cached
    def hessian(self):
        hh = self._Q_x_y("h", "h")
        ha = self._Q_x_y("h", "a")
        hb = self._Q_x_y("h", "b")
        aa = self._Q_x_y("a", "a")
        ab = self._Q_x_y("a", "b")
        bb = self._Q_x_y("b", "b")
        hess = np.array([[hh, ha, hb],
                         [ha, aa, ab],
                         [hb, ab, bb]])

        return np.sum(self.scaled_mass * hess, axis=2) / self.mean_scaling

    @cached
    def cov(self):
        return np.linalg.inv(-self.hessian)

    @cached
    def corr(self):
        s = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(s, s)


class MRP_PO_Likelihood_Weights(MRP_PO_Likelihood):
    """
    Effectively, this is the same as :class:`MRP_PO_Likelihood`, but instead of
    passing a full array of halo masses, one can pass an array of unique masses,
    and a `weights` array which specifies the number of each mass in the sample.

    This should be useful for simulation-like data, which has many halos of the
    same mass.
    """
    def __init__(self, weights, *args, **kwargs):
        super(MRP_PO_Likelihood_Weights, self).__init__(*args, **kwargs)
        self.weights = weights

    @cached
    def scaled_mass(self):
        return self.weights * self.m ** self.scale

    @cached
    def mean_scaling(self):
        return np.sum(self.scaled_mass) / np.sum(self.weights)



class MRP_Curve_Likelihood(MRP_PO_Likelihood):
    """
    A subclass of MRP_PO_Likelihood which implements Jacobians/Hessians for
    binned data (o theoretical curves).
    """
    def __init__(self,lnA,data,sig_rhomean=np.inf,sig_integ=np.inf,sig_data=1,Om0=0.3,
                 rhoc=2.7755e11,**kwargs):
        super(MRP_Curve_Likelihood,self).__init__(**kwargs)
        self.sig_rhomean=sig_rhomean
        self.sig_integ = sig_integ
        self.data = data
        self.Om0=Om0
        self.rhoc = rhoc
        self.sig_data=sig_data

        # Now correctly set alpha and A if necessary
        if sig_integ==0 and sig_rhomean==0:
            A,self.alpha = get_alpha_and_A(self.logHs,self.beta, self.log_mmin,
                                                self.mw_integ, Om0,self.scale,rhoc)
            self.lnA = np.log(A)
            self._alpha_s = self.alpha + self.scale

        elif sig_integ==0:
            self.lnA = np.log(self.mw_integ*self._q_) - self.scale*self.logHs*np.log(10)

        elif sig_rhomean==0:
            self.lnA = np.log(core.A_rhoc(self.logHs,self.alpha,self.beta,Om0,rhoc))
        else:
            self.lnA = lnA
    @cached
    def _zk(self):
        """
        The z-factor for the total integral, i.e.: (alpha+2)/beta.
        """
        return (self.alpha+2)/self.beta

    @cached
    def mw_integ(self):
        return simps(self.data*self.m**self.scale,self.m)

    #===========================================================================
    # Basic likelihood
    #===========================================================================
    @cached
    def Delta(self):
        """
        Distance between log theory and log data
        """
        return self.lnA + self._lng - np.log(self.data)

    @cached
    def delta_integ(self):
        return self.lnA - np.log(self.mw_integ/self._q_)+self.scale*self.logHs*ln10

    @cached
    def delta_rhomean(self):
        return self.lnA - np.log(core.A_rhoc(self.logHs,self.alpha,self.beta,self.Om0,self.rhoc))

    @cached
    def _lnLi(self):
        """
        Logarithmic likelihood of the samples
        """
        base = self.Delta**2/(2*self.sig_data**2)
        erri = 0
        errm = 0
        if (self.sig_rhomean==0 or np.isinf(self.sig_rhomean)) and (self.sig_integ==0 or np.isinf(self.sig_integ)):
            return -base
        else:
            if self.sig_rhomean!=0 and not np.isinf(self.sig_rhomean):
                errm = self.delta_rhomean**2/(2*self.sig_rhomean**2)
            if self.sig_integ!=0 and not np.isinf(self.sig_integ):
                erri = self.delta_integ**2/(2*self.sig_integ**2)
        return -(base+errm+erri)

    #===========================================================================
    # Simple lnk(theta) derivatives
    #===========================================================================
    @cached
    def _lnk_h(self):
        return 2*ln10

    @cached
    def _lnk_a(self):
        return polygamma(0,self._zk)/self.beta

    @cached
    def _lnk_b(self):
        return -self._lnk_a * self._zk

    #===========================================================================
    # Jacobian etc.
    #===========================================================================
    @cached
    def jacobian(self):
        base = self.Delta * np.array([self._lng_h,self._lng_a,self._lng_b,np.ones_like(self.m)])/self.sig_data**2
        errm = np.zeros(4)
        erri = np.zeros(4)
        if self.sig_integ !=0 and not np.isinf(self.sig_integ):
            erri = self.delta_integ*np.array([self._lnq_h+self.scale*ln10,self._lnq_a,self._lnq_b,1])/self.sig_integ**2
        if self.sig_rhomean != 0 and not np.isinf(self.sig_rhomean):
            errm = self.delta_rhomean*np.array([self._lnk_h,self._lnk_a,self._lnk_b,1])/self.sig_rhomean**2

        if self.sig_rhomean==0 and self.sig_integ==0:
            raise NotImplementedError()
            # return -np.sum(base,axis=1)[[0,2]]
        elif self.sig_rhomean==0:
            raise NotImplementedError()
            # return -np.sum(base[:-1].T + erri[:-1],axis=0)
        elif self.sig_integ==0:
            raise NotImplementedError()
            # return -np.sum(base[:-1].T + errm[:-1],axis=0)
        else:
            return -np.sum(base.T + erri + errm,axis=0)
