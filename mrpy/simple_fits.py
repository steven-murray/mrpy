"""
Routines that implement simple least-squares fits directly to dn/dm without errors.
Typically used for fitting directly to theoretical functions from EPS theory.
"""
import numpy as np
from core import mrp,pdf_norm,A_rhoc
from scipy.optimize import minimize
from scipy.integrate import simps

def get_fit_analytic(m,dndm,hs0=14.5,alpha0=-1.9,beta0=0.8,lnA0=-40,mmin=None,mmax=np.inf,
                     Om0=0.3,rhoc=2.7755e11,sigma_rhomean=np.inf,sigma_integ=np.inf,
                     s=1):
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
    if mmin is None:
        mmin = np.log10(m.min())

    mask = np.logical_and(m>=10**mmin,m<=10**mmax)

    res = fit_mrp_analytic(m,dndm,hs0,alpha0,beta0,lnA0,mmax,
                       Om0,rhoc,sigma_rhomean,sigma_integ,s)

    fit = mrp(m, res.x[0],res.x[1],res.x[2],mmin=mmin,mmax=mmax,norm=np.exp(res.x[3]))

    return res, fit

def fit_mrp_analytic(m,dndm,hs0=14.5,alpha0=-1.9,beta0=0.8,lnA0=-40,mmax=np.inf,
                 Om0=0.3,rhoc=2.7755e11,sigma_rhomean=np.inf,sigma_integ=np.inf,
                 s=1):
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
    """
    # If mmax is None, set to the maximum in m
    if mmax is None:
        mmax=np.log10(m.max())

    # For efficiency, take log of data and integral here.
    lndm = np.log(dndm)
    mass_weighted_integ = simps(dndm*m**s,m)

    ## Define functions to calculate error on the normalisation from rhomean
    ## or the data integral
    def err_rhomean(x):
        return x[-1] - np.log(A_rhoc(x[0],x[1],x[2],Om0,rhoc))

    def err_integ(x):
        return x[-1] + s*x[0]*np.log(10) - np.log(mass_weighted_integ*pdf_norm(x[0],x[1]+s,x[2],
                                                           mmin=np.log10(m.min()),mmax=mmax))

    ### Set up constraints if necessary.
    cons = ()
    if sigma_rhomean == 0 and sigma_integ==0:
        #This may not quite work very well
        cons = ({"type":"eq","fun":err_rhomean},
                {"type":"eq","fun":err_integ})
    elif sigma_rhomean == 0:
        cons = {"type":"eq","fun":err_rhomean}
    elif sigma_integ == 0:
        cons = {"type":"eq","fun":err_integ}

    ## Define the objective function for minimization.
    def model(p):
        d = mrp(m, p[0],p[1],p[2],norm=np.exp(p[3]),mmax=mmax,log=True)
        errm = 0
        erri = 0

        if sigma_rhomean !=0 and not np.isinf(sigma_rhomean):
            errm = err_rhomean(p)**2/sigma_rhomean**2

        if sigma_integ !=0 and not np.isinf(sigma_integ):
            erri = err_integ(p)**2/sigma_integ**2

        return np.sum((d-lndm)**2) + errm + erri

    p0 = [hs0, alpha0, beta0,lnA0]

    # There's a bit of hackery here, with the bounds. This should be fixed.
    # Also, we could provide the jacobian for faster fits.
    return  minimize(model, p0, method="SLSQP",constraints=cons,bounds=[(hs0-4,hs0+4),(max(-1.99,alpha0-0.3),alpha0+0.3),
                                                                        (beta0-0.5,beta0+0.5),(lnA0-20,lnA0+20)])
