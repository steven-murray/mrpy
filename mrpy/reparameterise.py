"""
Provides classes which implement variations of the MRP, in which the parameters have
been transformed.

Each transformation has three available classes:

    * one with a suffix `MRP`, which implements the core MRP quantities (i.e. is a subclass of :class:`mrpy.core.MRP`)
    * one with a suffix `PerObj`, which extends the previous one for likelihoods based on samples of variates (i.e. is a subclass of :class:`mrpy.likelihoods.PerObjLike`)
    * one with a suffix `Curve`, which extends the base one for likelihoods based on chi-squared minimzation against binned data (i.e. a subclass of :class:`mrpy.likelihoods.CurveLike`)

In all, the transformed parameters are denoted ``p1,p2,p3``.

In addition, base classes for each are provided, which makes it easy to implement
arbitrary transformations. See the docs for :class:`ReparameteriseMRP` for more
details.
"""

from cached_property import cached_property as _cached
import core
import likelihoods as lk
import numpy as np


class ReparameteriseMRP(core.MRP):
    """
    Base class for reparameterising the MRP.

    Reparameterisations take the form of a transformation of parameters. Two transformations are
    required, one from the new parameters, ``p1, p2, p3`` to the standard ones, ``logHs, alpha, beta``,
    and also the inverse of this transform.

    Actual reparameterisations should be based on this class and provide explicit functions for these
    transformations in the methods :func:`p_T` and :func:`theta_T`. This class provides the identity
    transforms.

    The class may be initialised either with the new parameters or standard ones, which is useful
    for different applications.

    Parameters
    ----------
    p1, p2, p3 : array_like, optional
        Values of the transformed parameters. Either all of these *or* all of the standard params
        must be provided. If both are provided, these are used.

    logHs, alpha, beta : array_like, optional
        Values of the un-transformed parameters. Either all of these *or* all of the transformed params
        must be provided. If both are provided, the transformed params are used.

    kwargs
        All of the other parameters are passed directly to the super-class.

    """
    # Step-size to use in numerical derivatives
    eps = 1e-6

    def __init__(self,p1=None,p2=None,p3=None, logHs=None,alpha=None,beta=None,**kwargs):
        if not p1 is None and not p2 is None and not p3 is None:
            self.p1, self.p2, self.p3 = p1,p2,p3
            kwargs["logHs"], kwargs["alpha"], kwargs['beta'] = self.theta_T()

        elif not logHs is None and not alpha is None and not beta is None:
            kwargs['logHs'] = logHs
            kwargs['alpha'] = alpha
            kwargs['beta'] = beta
        else:
            raise ValueError("Either all of p1,p2,p3 or all of logHs,alpha,beta must be specified.")

        super(ReparameteriseMRP, self).__init__(**kwargs)

        if not hasattr(self,"p1"):
            # If standard params passed, have to do this
            (self.p1, self.p2, self.p3) = self.p_T()

    def p_T(self, **kwargs):
        """
        The new parameters as functions of theta
        """
        return np.array([kwargs.get(s, getattr(self, s)) for s in ["logHs", "alpha", "beta"]])

    def theta_T(self, **kwargs):
        """
        theta as functions of the new parameters p1,p2,p3
        """
        return np.array([kwargs.get("p%s"%i, getattr(self, "p%s"%i)) for i in range(3)])


class ReparameterisePerObjLike(ReparameteriseMRP, lk.PerObjLike):
    """
    An extension of :class:`ReparameteriseMRP` which adds necessary methods for calculating
    jacobians and hessians for per-object likelihoods.

    See :class:`ReparameteriseMRP` for arguments.
    """
    def _dtheta_dp1(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        p1 = kwargs.pop("p1", self.p1)
        return (self.theta_T(p1=p1 + self.eps, **kwargs) - self.theta_T(p1=p1, **kwargs))/self.eps

    def _dtheta_dp2(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        p2 = kwargs.pop("p2", self.p2)
        return (self.theta_T(p2=p2 + self.eps, **kwargs) - self.theta_T(p2=p2, **kwargs))/self.eps

    def _dtheta_dp3(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        p3 = kwargs.pop("p3", self.p3)
        return (self.theta_T(p3=p3 + self.eps, **kwargs) - self.theta_T(p3=p3, **kwargs))/self.eps

    def _dtheta_dp1dp1(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        f0 = self._dtheta_dp1(**kwargs)
        p1 = kwargs.pop("p1", self.p1)
        return (self._dtheta_dp1(p1=p1 + self.eps, **kwargs) - f0)/self.eps

    def _dtheta_dp1dp2(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        f0 = self._dtheta_dp1(**kwargs)
        p1 = kwargs.pop("p1", self.p1)
        return (self._dtheta_dp2(p1=p1 + self.eps, **kwargs) - f0)/self.eps

    def _dtheta_dp1dp3(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        f0 = self._dtheta_dp1(**kwargs)
        p1 = kwargs.pop("p1", self.p1)
        return (self._dtheta_dp3(p1=p1 + self.eps, **kwargs) - f0)/self.eps

    def _dtheta_dp2dp2(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        f0 = self._dtheta_dp2(**kwargs)
        p2 = kwargs.pop("p2", self.p2)
        return (self._dtheta_dp2(p2=p2 + self.eps, **kwargs) - f0)/self.eps

    def _dtheta_dp2dp3(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        f0 = self._dtheta_dp2(**kwargs)
        p2 = kwargs.pop("p2", self.p2)
        return (self._dtheta_dp3(p2=p2 + self.eps, **kwargs) - f0)/self.eps

    def _dtheta_dp3dp3(self, **kwargs):
        """
        Replace this with analytic function if possible!
        """
        f0 = self._dtheta_dp3(**kwargs)
        p3 = kwargs.pop("p3", self.p3)
        return (self._dtheta_dp3(p3=p3 + self.eps, **kwargs) - f0)/self.eps

    @property
    def this_jacobian(self):
        """Jacobian of the reparameterisation."""
        j = self.jacobian
        return np.array([np.dot(j, self._dtheta_dp1()), np.dot(j, self._dtheta_dp2()),
                         np.dot(j, self._dtheta_dp3())])

    @property
    def this_hessian(self):
        """Hessian of the reparameterisation."""
        # jac and hess of lnL with MRP params
        j = self.jacobian
        H = self.hessian

        # jacs of transformation
        j1 = self._dtheta_dp1()
        j2 = self._dtheta_dp2()
        j3 = self._dtheta_dp3()

        h11 = np.dot(j, self._dtheta_dp1dp1()) + np.dot(j1, np.dot(H, j1))
        h12 = np.dot(j, self._dtheta_dp1dp2()) + np.dot(j1, np.dot(H, j2))
        h13 = np.dot(j, self._dtheta_dp1dp3()) + np.dot(j1, np.dot(H, j3))
        h22 = np.dot(j, self._dtheta_dp2dp2()) + np.dot(j2, np.dot(H, j2))
        h23 = np.dot(j, self._dtheta_dp2dp3()) + np.dot(j2, np.dot(H, j3))
        h33 = np.dot(j, self._dtheta_dp3dp3()) + np.dot(j3, np.dot(H, j3))

        out = np.array([[h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])
        return out

    @property
    def this_cov(self):
        """Covariance matrix of the reparameterisation,"""
        return np.linalg.inv(-self.this_hessian)

    @property
    def cov_ratio(self):
        """Ratio of the reparameterised covariance to the standard covariance."""
        return self.this_cov/self.cov

    @property
    def hess_ratio(self):
        """"Ratio of the reparameterised hessian to the standard hessian."""
        return self.this_hessian/self.hessian

    @property
    def this_corr(self):
        """Correlation matrix of the reparameterisation."""
        cov = self.this_cov
        s = np.sqrt(np.diag(cov))
        return cov/np.outer(s, s)

    @property
    def corr_ratio(self):
        """Ratio of the reparameterised correlation to the standard correlation."""
        return self.this_corr/self.corr


class ReparameteriseCurveLike(ReparameterisePerObjLike, lk.CurveLike):
    """
    An extension of :class:`ReparameteriseMRP` which adds necessary methods for calculating
    jacobians and hessians for chi-square likelihoods.

    See :class:`ReparameteriseMRP` for arguments.
    """
    pass


# ===========================================================================================
# Ap1
# ===========================================================================================
class _Ap1MRP(ReparameteriseMRP):
    """
    A fairly standard parameterisation of the TGGD (eg. Lagos, Ferreira, Valenzuela Hube, 2011,
    Eq. 2.1)

    This takes advantage of the base class numerical derivatives.

    Notes
    -----
    This parameterisation is determined by setting

    .. math:: (\log \mathcal{H}_\star, \alpha, \beta) = (p_1, p_2-1, p_3).
    """

    def p_T(self, **kwargs):
        """
        New parameters as functions of the standard MRP parameters.
        """
        return np.array([self.logHs, self.alpha + 1, self.beta])

    def theta_T(self, p1=None, p2=None, p3=None):
        """
        Standard MRP parameters as functions of the new parameters.
        """
        p1 = p1 or self.p1
        p2 = p2 or self.p2
        p3 = p3 or self.p3
        return np.array([p1, p2 - 1, p3])


class _AP1PerObj(_Ap1MRP, ReparameterisePerObjLike):
    pass


class _AP1Curve(_Ap1MRP, ReparameteriseCurveLike):
    pass


class Ap1MRP(_Ap1MRP):
    """
    A fairly standard parameterisation of the TGGD (eg. Lagos et al. Eq. 2.1)

    Notes
    -----
    This parameterisation is determined by setting

    .. math:: (\log \mathcal{H}_\star, \alpha, \beta) = (p_1, p_2-1, p_3).
    """

    def _dtheta_dp1(self, **kwargs):
        return np.array([1, 0, 0])

    def _dtheta_dp2(self, **kwargs):
        return np.array([0, 1, 0])

    def _dtheta_dp3(self, **kwargs):
        return np.array([0, 0, 1])

    def _dtheta_dp1dp1(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp3(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp2dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp2dp3(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp3dp3(self, **kwargs):
        return np.array([0, 0, 0])


class AP1PerObj(Ap1MRP, ReparameterisePerObjLike):
    pass


class AP1Curve(Ap1MRP, ReparameteriseCurveLike):
    pass


# ===========================================================================================
# GG2
# ===========================================================================================
class GG2MRP(ReparameteriseMRP):
    """
    A reparameterisation of the standard MRP form of the TGGD (eg. Lagos et al. Eq. 2.3)

    Notes
    -----
    This has the form

    .. math:: (\log \mathcal{H}_\star, \alpha, \beta) = (-p_1/p_3, p_2-1, p_3).
    """

    def p_T(self, **kwargs):
        """
        New parameters as functions of the standard MRP parameters.
        """
        return np.array([-self.beta*self.logHs, self.alpha + 1, self.beta])

    def theta_T(self, p1=None, p2=None, p3=None):
        """
        Standard MRP parameters as functions of the new parameters.
        """
        p1 = p1 or self.p1
        p2 = p2 or self.p2
        p3 = p3 or self.p3
        return np.array([-p1/p3, p2 - 1, p3])


class GG2PerObj(GG2MRP, ReparameterisePerObjLike):
    def _dtheta_dp1(self, **kwargs):
        p3 = kwargs.get("p3", self.p3)
        return np.array([-1./p3, 0, 0])

    def _dtheta_dp2(self, **kwargs):
        return np.array([0, 1, 0])

    def _dtheta_dp3(self, **kwargs):
        p3 = kwargs.get("p3", self.p3)
        p1 = kwargs.get("p1", self.p1)
        return np.array([p1/p3**2, 0, 1])

    def _dtheta_dp1dp1(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp3(self, **kwargs):
        p3 = kwargs.get("p3", self.p3)
        return np.array([1.0/p3**2, 0, 0])

    def _dtheta_dp2dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp2dp3(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp3dp3(self, **kwargs):
        p3 = kwargs.get("p3", self.p3)
        p1 = kwargs.get("p1", self.p1)
        return np.array([-2*p1/p3**3, 0, 0])


class GG2Curve(GG2PerObj, ReparameteriseCurveLike):
    pass


# ===========================================================================================
# GG3
# ===========================================================================================
class GG3MRP(ReparameteriseMRP):
    """
    A reparameterisation of the standard MRP form of the TGGD (eg. Lagos et al. Eq. 2.4)

    Notes
    -----
    This has the form

    .. math:: (\log \mathcal{H}_\star, \alpha, \beta) = (p_1, p_2 p_3-1, p_3).
    """

    def p_T(self, **kwargs):
        """
        New parameters as functions of the standard MRP parameters.
        """
        return np.array([self.logHs, (self.alpha + 1)/self.beta, self.beta])

    def theta_T(self, p1=None, p2=None, p3=None):
        """
        Standard MRP parameters as functions of the new parameters.
        """
        p1 = p1 or self.p1
        p2 = p2 or self.p2
        p3 = p3 or self.p3
        return np.array([p1, p2*p3 - 1, p3])


class GG3PerObj(GG3MRP, ReparameterisePerObjLike):
    def _dtheta_dp1(self, **kwargs):
        return np.array([1.0, 0, 0])

    def _dtheta_dp2(self, **kwargs):
        p3 = kwargs.get("p3", self.p3)
        return np.array([0, p3, 0])

    def _dtheta_dp3(self, **kwargs):
        p2 = kwargs.get("p2", self.p2)
        return np.array([0, p2, 1])

    def _dtheta_dp1dp1(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp3(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp2dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp2dp3(self, **kwargs):
        return np.array([0, 1.0, 0])

    def _dtheta_dp3dp3(self, **kwargs):
        return np.array([0, 0, 0])


class GG3Curve(GG3PerObj, ReparameteriseCurveLike):
    pass


# ===========================================================================================
# HT
# ===========================================================================================
class HTMRP(ReparameteriseMRP):
    """
    A reparameterisation of the standard MRP form of the TGGD.

    Notes
    -----
    This has the form

    .. math:: (\log \mathcal{H}_\star, \alpha, \beta) = (p_1 - \log_{10}(\frac{2+p_2}{p_3}),p_2,p_3),

    where the transformation of :math:`\log \mathcal{H}_\star` is to the logarithmic mass mode.
    """

    def p_T(self, **kwargs):
        """
        New parameters as functions of the standard MRP parameters.
        """
        return np.array([self.logHs + np.log10((2 + self.alpha)/self.beta)/self.beta, self.alpha, self.beta])

    def theta_T(self, p1=None, p2=None, p3=None):
        """
        Standard MRP parameters as functions of the new parameters.
        """
        p1 = p1 or self.p1
        p2 = p2 or self.p2
        p3 = p3 or self.p3
        return np.array([p1 - np.log10((2 + p2)/p3)/p3, p2, p3])


class HTPerObj(HTMRP, ReparameterisePerObjLike):
    def _dtheta_dp1(self, **kwargs):
        return np.array([1.0, 0, 0])

    def _dtheta_dp2(self, **kwargs):
        p2 = kwargs.get("p2", self.p2)
        p3 = kwargs.get("p3", self.p3)
        return np.array([-1./(np.log(10)*p3*(2 + p2)), 1.0, 0])

    def _dtheta_dp3(self, **kwargs):
        p2 = kwargs.get("p2", self.p2)
        p3 = kwargs.get("p3", self.p3)
        return np.array([(np.log((2 + p2)/p3) + 1)/(p3**2*np.log(10)), 0, 1])

    def _dtheta_dp1dp1(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp2(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp1dp3(self, **kwargs):
        return np.array([0, 0, 0])

    def _dtheta_dp2dp2(self, **kwargs):
        p2 = kwargs.get("p2", self.p2)
        p3 = kwargs.get("p3", self.p3)
        return np.array([1./(p3*(2 + p2)**2*np.log(10)), 0, 0])

    def _dtheta_dp2dp3(self, **kwargs):
        p2 = kwargs.get("p2", self.p2)
        p3 = kwargs.get("p3", self.p3)
        return np.array([1./(p3**2*(2 + p2)*np.log(10)), 0, 0])

    def _dtheta_dp3dp3(self, **kwargs):
        p2 = kwargs.get("p2", self.p2)
        p3 = kwargs.get("p3", self.p3)
        return np.array([-(2*np.log((2 + p2)/p3) + 3)/(p3**3*np.log(10)), 0, 0])


class HTCurve(HTPerObj, ReparameteriseCurveLike):
    pass