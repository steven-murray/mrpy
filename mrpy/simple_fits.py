"""
Routines that implement simple least-squares fits directly to dn/dm without errors.
Typically used for fitting directly to theoretical functions from EPS theory.
"""
import numpy as np
from core import mrp,pdf_norm,A_rhoc, get_alpha_and_A
from scipy.optimize import minimize
from scipy.integrate import simps
from special import gamma, gammainc
from scipy.optimize import newton

#TODO: better initial guesses, better bounds, jacobians

def get_fit_analytic(*args,**kwargs):
    """
    Wrapper for fit_mrp_analytic, returning the fitted parameters along with fitted curve.

    All parameters the same as :func:`fit_mrp_analytic`.

    Returns
    -------
    res : scipy.optiimize.minimize result
        The result of the minimization. Access the parameters with ``res.x``.

    fit : array
        Resulting fitted mass function over ``m``.
    """
    res = fit_mrp_analytic(*args,**kwargs)

    if len(args) > 0:
        m = args[0]
    else:
        m = kwargs['m']

    fit = mrp(m, res[0],res[1],res[2],mmin=np.log10(m.min()),mmax=np.log10(m.max()),
              norm=np.exp(res[3]))
    return res, fit

def fit_mrp_analytic(m,dndm,hs0=14.5,alpha0=-1.9,beta0=0.8,lnA0=-40,mmax=np.inf,
                 Om0=0.3,rhoc=2.7755e11,sigma_rhomean=np.inf,sigma_integ=np.inf,
                 s=1,bounds=True,hs_bounds=(0,16),alpha_bounds=(-2,-1.3),
                 beta_bounds=(0.1,5.0),lnA_bounds=(-50,0),jac=True,**minimize_kw):
    """
    Basic LSQ fit for the MRP parameters, with flexible constraints.

    By default, all four parameters are fit to the data at hand.
    However, ``sigma_rhomean`` and ``sigma_integ`` control the
    uncertainties on the knowledge of rhomean and the integral of the data. By
    default they are infinite, and therefore unused. If either is set to 0, the
    normalisation is completely fixed. If *both* are set to 0, the normalisation
    *and* alpha are fixed.

    Parameters
    ----------
    m : array
        Masses

    dndm : array
        Mass function corresponding to m.

    hs0, alpha0, beta0, lnA0 : float, optional
        Initial guess for each of the MRP parameters.

    mmax : float, optional
        Log-10 maximum mass to use to calculate the pdf. Should be either ``inf`` or
        the log-10 maximum mass in ``m``, set with ``None``. Using ``inf`` is
        slightly more efficient, and should be correct if the maximum mass is
        high enough.

    Om0 : float, optional
        Mean matter density parameter

    rhoc : float, optional
        The critical density of the universe.

    sigma_rhomean : float, optional
        Controls the uncertainty on the mean density of the Universe.
        Set as ``inf`` to not consider it at all, and zero to render it a
        perfect constraint. In between, it technically represents the uncertainty
        on ``rho_mean/k(theta)``.

    sigma_integ : float, optional
        Controls the uncertainty on the (mass-weighted) integral of the data.
        Set as ``inf`` to not consider it at all, and zero to render it a
        perfect constraint. In between, it technically represents the uncertainty
        on ``int.*q(theta)``.

    s : float, optional
        Mass scaling. This is used *only* for the mass-weighted integral of the
        data, which influences the constraint from ``sigma_integ``.

    bounds : None or True
        If None, don't use bounds. If true, set bounds based on bounds passed.

    hs_bounds, alpha_bounds, beta_bounds, lnA_bounds : 2-tuple
        2-tuples specifying minimum and maximum values for each bound.

    jac : bool, optional
        Whether to use analytic jacobian (usually a good idea)

    minimize_kw : dict
        Any other parameters to :func:`scipy.minimize`.
    """
    # If mmax is None, set to the maximum in m
    if mmax is None:
        mmax=np.log10(m.max())

    # For efficiency, take log of data and do integral here.
    lndm = np.log(dndm)
    mw_data = dndm*m**s
    mass_weighted_integ = simps(mw_data,m)

    ## Define the objective function for minimization.
    def model(p):
        if len(p) == 4:
            hs,alpha,beta,lnA = p
        elif len(p)==3:
            lnA = 0
            hs,alpha,beta = p
        else:
            lnA = 0
            alpha = 0
            hs,beta = p

        _curve = MRP_Curve_Likelihood(logm=log10(m),logHs=hs,alpha=alpha,beta=beta,lnA=lnA,
                                      sig_rhomean=sigma_rhomean,sig_integ=sigma_integ,scale=s,mw_data=mw_data,
                                      mw_integ = mass_weighted_integ)
        if jac:
            return -_curve.lnL,-_curve.jacobian
        else:
            return -_curve.lnL



    if sigma_integ==0 and sigma_rhomean==0:
        p0 = [hs0,beta0]
        if bounds:
            bounds = [hs_bounds,beta_bounds]
        res = minimize(model, p0, bounds=bounds,jac=jac,**minimize_kw)
        _c = MRP_Curve_Likelihood(logm=log10(m),logHs=res.x[0],alpha=res.x[1],beta=res.x[2],lnA=0,
                                      sig_rhomean=sigma_rhomean,sig_integ=sigma_integ,scale=s,mw_data=mw_data,
                                      mw_integ = mass_weighted_integ)
        lnA = _c.lnA
        alpha = _c.alpha

        out = [res.x[0],alpha,res.x[1],lnA]

    elif sigma_integ==0:
        p0 = [hs0, alpha0, beta0]
        if bounds: bounds = [hs_bounds,alpha_bounds,beta_bounds]
        res = minimize(model, p0, bounds=bounds,jac=jac,**minimize_kw)
        out = np.concatenate((res.x,[MRP_Curve_Likelihood(logm=log10(m),logHs=res.x[0],alpha=res.x[1],beta=res.x[2],lnA=0,
                                      sig_rhomean=sigma_rhomean,sig_integ=sigma_integ,scale=s,mw_data=mw_data,
                                      mw_integ = mass_weighted_integ).lnA]))

    elif sigma_rhomean==0:
        p0 = [hs0, alpha0, beta0]
        if bounds: bounds = [hs_bounds,alpha_bounds,beta_bounds]
        res = minimize(model, p0, bounds=bounds,jac=jac,**minimize_kw)
        out = np.concatenate((res.x ,[MRP_Curve_Likelihood(logm=log10(m),logHs=res.x[0],alpha=res.x[1],beta=res.x[2],lnA=0,
                                      sig_rhomean=sigma_rhomean,sig_integ=sigma_integ,scale=s,mw_data=mw_data,
                                      mw_integ = mass_weighted_integ).lnA]))

    else:
        p0 = [hs0, alpha0, beta0,lnA0]
        if bounds: bounds = [hs_bounds,alpha_bounds,beta_bounds,lnA_bounds]
        res = minimize(model, p0, bounds=bounds,jac=jac,**minimize_kw)
        out = res.x

    return out
