"""
Routines that implement simple least-squares fits directly to dn/dm without errors.
Typically used for fitting directly to theoretical functions from EPS theory.

For more involved fits to non-binned data, see :module:`mrpy.fit_perobj`. For the
definition of the likelihood involved in the fits within this module see
:class:`mrpy.likelihoods.CurveLike`.
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as intg
import likelihoods as lk


def get_fit_curve(m, dndm, hs0=14.5, alpha0=-1.9, beta0=0.8, lnA0=-40,
              Om0=0.3, rhoc=2.7755e11, sigma_rhomean=np.inf, sigma_integ=np.inf,
              s=0, bounds=True, hs_bounds=(0, 16), alpha_bounds=(-2, -1.3),
              beta_bounds=(0.1, 5.0), lnA_bounds=(-50, 0), jac=True, **minimize_kw):
    """
    Basic LSQ fit for the MRP parameters, with flexible constraints.

    By default, all four parameters are fit to the data at hand.
    However, ``sigma_rhomean`` and ``sigma_integ`` control the
    uncertainties on the knowledge of `rhomean` and the integral of the data. By
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

    Om0 : float, optional
        Mean matter density parameter. Only used if `sigma_rhomean` is not ``inf``.

    rhoc : float, optional
        The critical density of the universe. Only used if `sigma_rhomean` is not ``inf``.

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

    curve : :class:`mrpy.likelihoods.CurveLike` object
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

    Examples
    --------
    The most obvious example is to generate a mass function curve from given parameters:

    >>> from mrpy import dndm
    >>> import numpy as np
    >>> m = np.logspace(10,15,500)
    >>> d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    Then find the best-fit parameters for the resulting data:

    >>> from mrpy import get_fit_curve
    >>> res,curve = get_fit_curve(m,d,lnA0=0.0)
    >>> print res.x

    If the total integral of the curve is required to integrate to the total
    mean density of the Universe, then we can set this as a constraint by using

    >>> d = dndm(m,14.0,-1.9,0.75,norm="rhoc")
    >>> res, curve = get_fit_curve(m,d,sigma_rhomean=0.0,hs_bounds=(10,16),alpha_bounds=(-1.95,-1.5))
    >>> print res.x

    We can also use the ``curve`` object to explore some of the qualities of the fit

    >>> print curve.hessian
    >>> from matplotlib.pyplot import plot
    >>> plot(curve.logm,curve.dndm(log=True))
    >>> plot(curve.logm,np.log(d))
    >>> print curve.stats.mean

    """
    # For efficiency, take log of data and do integral here.
    lndm = np.log(dndm)
    mw_data = dndm*m**s
    mass_weighted_integ = intg.simps(mw_data, m)

    # Define the objective function for minimization.
    def model(p):
        if len(p) == 4:
            hs, alpha, beta, lnA = p
        elif len(p) == 3:
            lnA = 0
            hs, alpha, beta = p
        else:
            lnA = 0
            alpha = 0
            hs, beta = p

        _curve = lk.CurveLike(logm=np.log10(m), logHs=hs, alpha=alpha, beta=beta, norm=lnA,
                           sig_rhomean=sigma_rhomean, sig_integ=sigma_integ, scale=s,
                           mw_data=mw_data, mw_integ=mass_weighted_integ)
        if jac:
            return -_curve.lnL, -_curve.jacobian
        else:
            return -_curve.lnL

    if sigma_integ == 0 and sigma_rhomean == 0:
        raise NotImplementedError()

        # p0 = [hs0, beta0]
        # if bounds:
        #     bounds = [hs_bounds, beta_bounds]
        # res = minimize(model, p0, bounds=bounds, jac=jac, **minimize_kw)
        # _c =CurveLike(logm=np.log10(m), logHs=res.x[0], alpha=res.x[1], beta=res.x[2], lnA=0,
        #               sig_rhomean=sigma_rhomean, sig_integ=sigma_integ, scale=s, mw_data=mw_data,
        #               mw_integ=mass_weighted_integ)
        # lnA = _c.lnA
        # alpha = _c.alpha
        #
        # out = [res.x[0], alpha, res.x[1], lnA]

    elif sigma_integ == 0 or sigma_rhomean == 0:
        p0 = [hs0, alpha0, beta0]
        if bounds: bounds = [hs_bounds, alpha_bounds, beta_bounds]
        # res = minimize(model, p0, bounds=bounds, jac=jac, **minimize_kw)
    #     # out = np.concatenate(
    #     #     (res.x, [CurveLike(logm=np.log10(m), logHs=res.x[0], alpha=res.x[1], beta=res.x[2], norm=0,
    #     #                        sig_rhomean=sigma_rhomean, sig_integ=sigma_integ, scale=s, mw_data=mw_data,
    #     #                        mw_integ=mass_weighted_integ).lnA]))
    #
    # elif sigma_rhomean == 0:
    #     p0 = [hs0, alpha0, beta0]
    #     if bounds: bounds = [hs_bounds, alpha_bounds, beta_bounds]
    #     res = minimize(model, p0, bounds=bounds, jac=jac, **minimize_kw)
    #     # out = np.concatenate(
    #     #     (res.x, [CurveLike(logm=np.log10(m), logHs=res.x[0], alpha=res.x[1], beta=res.x[2], norm=0,
    #     #                        sig_rhomean=sigma_rhomean, sig_integ=sigma_integ, scale=s, mw_data=mw_data,
    #     #                        mw_integ=mass_weighted_integ).lnA]))

    else:
        p0 = [hs0, alpha0, beta0, lnA0]
        if bounds: bounds = [hs_bounds, alpha_bounds, beta_bounds, lnA_bounds]

    res = opt.minimize(model, p0, bounds=bounds, jac=jac, **minimize_kw)

    lnA = None
    if len(res.x) == 4:
        lnA = res.x[3]

    c = lk.CurveLike(logm=np.log10(m), logHs=res.x[0], alpha=res.x[1], beta=res.x[2], norm=lnA,
                  sig_rhomean=sigma_rhomean, sig_integ=sigma_integ, scale=s, mw_data=mw_data,
                  mw_integ=mass_weighted_integ)

    return res, c
