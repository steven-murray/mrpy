"""
Routines that implement fits directly to samples of masses (or other values
drawn from a TGGD).

For fits to binned data, see :module:`mrpy.fit_curve`. For the
definition of the likelihood involved in the fits within this module see
:class:`mrpy.likelihoods.PerObjLike`.
"""

import numpy as np
from scipy.optimize import minimize
from likelihoods import PerObjLike
from scipy.stats import truncnorm
import emcee

def get_fit_perobj(m, hs0=14.5, alpha0=-1.9, beta0=0.8,
                   Om0=0.3, rhoc=2.7755e11,s=0, bounds=True,
                   hs_bounds=(10, 16), alpha_bounds=(-1.99, -1.3),
                   beta_bounds=(0.1, 2.0), jac=True, **minimize_kw):
    """
    Per-object downhill fit for masses m.

    Parameters
    ----------
    m : array
        Masses

    hs0, alpha0, beta0: float, optional
        Initial guess for each of the MRP parameters.

    s : float, optional
        Mass scaling. Setting this greater than 0 upweights higher valued
        variates, `m`, with a weight of ``m**s``.

    bounds : None or True
        If None, don't use bounds. If true, set bounds based on bounds passed.

    hs_bounds, alpha_bounds, beta_bounds : 2-tuple
        2-tuples specifying minimum and maximum values for each bound.

    jac : bool, optional
        Whether to use analytic jacobian (usually a good idea)

    minimize_kw : dict
        Any other parameters to :func:`scipy.minimize`.

    Returns
    -------
    res : `OptimizeResult`
        The optimization result represented as a ``OptimizeResult`` object
        (see scipy documentation).
        Important attributes are: ``x`` the solution array, ``success`` a Boolean flag
        indicating if the optimizer exited successfully and ``message`` which describes
        the cause of the termination.

        The parameters are ordered by `logHs`, `alpha`, `beta`, `[lnA]`.

    perobj : :class:`mrpy.likelihoods.PerObjLike` object
        An object containing the solution parameters and methods to access
        relevant quantities, such as the mass function, or jacobian and
        hessian at the solution.

    Notes
    -----
    Though the option to *not* use bounds for the fit is available, at this point
    it seems to yield unpredictable results. Unless the problem at hand is so poorly
    known as to be impossible to set appropriate bounds, it is encouraged to use them.

    Furthermore, use as stringent bounds as possible, since the algorithm explores the
    edges, which can induce numerical error if values far from the solution are chosen.

    The setting of `s` can be tricky. It is sometimes necessary to set it higher than 0
    to achieve reasonable precision on `beta`, due to the severe relative lack of high-value
    variates. Nevertheless, doing so in general decreases the reliability of the fit. Simple
    tests show that in typical cases, about 4 times as many variates are required for an ``s=1``
    fit to achieve the same accuracy on parameters.

    Examples
    --------
    The most obvious example is to generate a sample of variates from given parameters:

    >>> from mrpy.stats import TGGD
    >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)

    Then find the best-fit parameters for the resulting data:

    >>> from mrpy.fit_perobj import get_fit_perobj
    >>> res,obj = get_fit_perobj(r)
    >>> print res.x

    We can also use the ``obj`` object to explore some of the qualities of the fit

    >>> print obj.hessian
    >>> print obj.cov
    >>> from matplotlib.pyplot import plot
    >>> plot(obj.logm,obj.dndm(log=True))
    >>> print obj.stats.mean, r.mean()
    """

    # Define the objective function for minimization.
    def model(p):
        _mod = PerObjLike(scale=s, logm=np.log10(m), logHs=p[0], alpha=p[1], beta=p[2])

        if jac:
            return -_mod.lnL, -_mod.jacobian
        else:
            return -_mod.lnL

    p0 = [hs0, alpha0, beta0]
    if bounds:
        bounds = [hs_bounds, alpha_bounds, beta_bounds]
    res = minimize(model, p0, bounds=bounds, jac=jac, **minimize_kw)

    perobj = PerObjLike(scale=s, logm=np.log10(m), logHs=res.x[0], alpha=res.x[1],
                        beta=res.x[2])

    return res, perobj


# =========================================================================================
# EMCEE BASED ROUTINES
# =========================================================================================
def _get_cuts(bound,mu,sigma):
    return (bound-mu)/sigma


def _get_initial_ball(guess,chains,hs_bounds,alpha_bounds,beta_bounds):
    s = 0.05
    a = np.array([hs_bounds[0], alpha_bounds[0], beta_bounds[0]])
    b = np.array([hs_bounds[1], alpha_bounds[1], beta_bounds[1]])

    a = _get_cuts(a, guess, np.abs(s * guess))
    b = _get_cuts(b, guess, np.abs(s * guess))
    stacked_val = np.empty((chains, len(guess)))
    for i, (g, A, B) in enumerate(zip(guess, a, b)):
        stacked_val[:, i] = truncnorm(A, B, loc=g, scale=np.abs(s * g)).rvs(chains)

    return stacked_val

def fit_perobj_emcee(m, nchains=50,warmup=1000,iterations=1000,
                     hs0=14.5, alpha0=-1.9, beta0=0.8,
                     Om0=0.3, rhoc=2.7755e11,
                     s=0, bounds=True, hs_bounds=(12, 16), alpha_bounds=(-1.99, -1.3),
                     beta_bounds=(0.3, 2.0),opt_init=False,opt_kw={},
                     prior_func=None,debug=0,
                     **kwargs):
    """
    Per-object MCMC fit for masses `m`, using the `emcee` package.

    Parameters
    ----------
    m : array
        Masses

    nchains : int, optional
        Number of chains to use in the AIES MCMC algorithm

    warmup : int, optional
        Number (discarded) warmup iterations.

    iterations : int, optional
        Number of iterations to keep in the chain.

    hs0, alpha0, beta0: float, optional
        Initial guess for each of the MRP parameters.

    s : float, optional
        Mass scaling. Setting this greater than 0 upweights higher valued
        variates, `m`, with a weight of ``m**s``.

    bounds : None or True
        If None, don't use bounds. If true, set bounds based on bounds passed.

    hs_bounds, alpha_bounds, beta_bounds : 2-tuple
        2-tuples specifying minimum and maximum values for each bound.

    opt_init : bool, optional
        Whether to run a downhill optimization routine to get the best
        starting point for the MCMC.

    opt_kw : dict, optional
        Any arguments to pass to the downhill run.

    prior_func : function, optional
        A function to calculate the likelihood arising from priors on the parameters.
        By default, uniform priors are assumed, which add nothing to the likelihood.
        The only parameter taken by the function should be the vector of parameter
        values, ``[logHs,alpha,beta]``.

    debug : int, optional
        Set the level of info printed out throughout the function. Highest current
        level that is useful is 2.

    kwargs :
        Any other parameters to :class:`emcee.EnsembleSampler`.

    Returns
    -------
    mcmc_res : :class:`emcee.EnsembleSampler` object
        This object contains the stored chains, and other attributes.

    Notes
    -----
    The setting of `s` can be tricky. It is sometimes necessary to set it higher than 0
    to achieve reasonable precision on `beta`, due to the severe relative lack of high-value
    variates. Nevertheless, doing so in general decreases the reliability of the fit. Simple
    tests show that in typical cases, about 4 times as many variates are required for an ``s=1``
    fit to achieve the same accuracy on parameters.

    Examples
    --------
    The most obvious example is to generate a sample of variates from given parameters:

    >>> from mrpy.stats import TGGD
    >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)

    Then find the best-fit parameters for the resulting data:

    >>> from mrpy.fit_perobj import fit_perobj_emcee
    >>> res = fit_perobj_emcee(r,nchains=10,warmup=100,iterations=100)
    >>> print res.flatchain.mean(axis=0)
    """
    # First, set guess, either by optimization or passed values
    guess = np.array([hs0,alpha0,beta0])
    if opt_init:
        res = get_fit_perobj(m,hs0,alpha0,beta0,Om0,rhoc,s,bounds,hs_bounds,alpha_bounds,
                               beta_bounds,**opt_kw)[0]
        if res.success:
            guess = res.x
            if debug:
                print "Optimization result (used as guess): ", guess
        else:
            print "WARNING: Optimization failed. Falling back on given guess."

    initial = _get_initial_ball(guess,nchains,hs_bounds,alpha_bounds,beta_bounds)


    # Define the likelihood function
    def model(p):
        # Some absolute bounds
        if p[2] < 0 or p[0] < 0:
            if debug>0:
                print "OUT OF BOUNDS: ", p
            return -np.inf

        # Enforced bounds
        if not hs_bounds[0] <= p[0] <= hs_bounds[1]:
            if debug>0:
                print "logHs out of bounds: ", p
            return - np.inf
        if not alpha_bounds[0] <= p[1] <= alpha_bounds[1]:
            if debug:
                print "alpha out of bounds: ", p
            return - np.inf
        if not beta_bounds[0] <= p[2] <= beta_bounds[1]:
            if debug:
                print "beta out of bounds: ", p
            return - np.inf

        # Priors
        if prior_func is None: #default uniform prior.
            ll = 0
        else:
            ll = prior_func(p)

        # Likelihood
        _mod = PerObjLike(scale=s, logm=np.log10(m), logHs=p[0], alpha=p[1], beta=p[2])
        ll += _mod.lnL

        if debug > 1:
            print "pars, ll: ", p, ll

        if np.isnan(ll):
            return -np.inf
        else:
            return _mod.lnL

    mcmc_res = emcee.EnsembleSampler(nchains,initial.shape[1],model,**kwargs)

    if warmup:
        initial, lnprob, rstate = mcmc_res.run_mcmc(initial, warmup, storechain=False)
        mcmc_res.reset()

    mcmc_res.run_mcmc(initial, iterations)

    return mcmc_res