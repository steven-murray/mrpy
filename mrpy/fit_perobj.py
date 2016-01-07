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

def get_fit_perobj(m, hs0=14.5, alpha0=-1.9, beta0=0.8, lnA0=-40,
                   Om0=0.3, rhoc=2.7755e11, sigma_rhomean=np.inf, sigma_integ=np.inf,
                   s=1, bounds=True, hs_bounds=(0, 16), alpha_bounds=(-2, -1.3),
                   beta_bounds=(0.1, 5.0), jac=True, **minimize_kw):
    """
    Per-object downhill fit for masses m.

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

    hs0, alpha0, beta0, lnA0 : float, optional
        Initial guess for each of the MRP parameters.

    mmax : float, optional
        Log-10 maximum mass to use to calculate the pdf. Should be either ``inf`` or
        the log-10 maximum mass in ``m``, set with ``None``. Using ``inf`` is
        slightly more efficient, and should be correct if the maximum mass is
        high enough.

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
