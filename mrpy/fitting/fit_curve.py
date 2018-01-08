"""
Routines that implement simple least-squares fits directly to dn/dm without errors.
Typically used for fitting directly to theoretical functions from EPS theory.

For more involved fits to non-binned data, see :module:`mrpy.fit_perobj`. For the
definition of the likelihood involved in the fits within this module see
:class:`mrpy.likelihoods.CurveLike`.
"""
import numpy as np
import scipy.optimize as opt

import mrpy.extra.likelihoods as lk
from mrpy import MRP

def get_fit_curve(data_m, data_mf, x0,
                  bounds=None, jac=True,
                  sig_rhomean=np.inf, sig_data = 1,
                  **minimize_kw):
    """
    Perform basic LSQ fitting of the MRP curve to the given curve. The fit is performed in log-log space

    Parameters
    ----------
    data_m : array
        Masses (not log)

    data_mf : array
        Mass function corresponding to m.

    x0 : float, optional
        Initial guess for each of the MRP parameters (hs, alpha, beta, lnA)

    bounds : list of tuples, optional
        If None, don't use bounds. If true, set bounds based on bounds passed.

    jac : bool, optional
        Whether to use analytic jacobian (usually a good idea)

    sig_data : array_like, optional
        The uncertainty of the data (standard deviation). This is used in the likelihood
        to weight different mass scales. If scalar, all mass scales are weighted evenly.

    sig_rhomean,: float, optional
        This controls how much influence the total mean density of the universe has
        on the likelihood. The default value of `inf` means it is completely ignored.
        If it is 0, it becomes an absolute constraint, so that the total mass density
        of the universe is perfectly matched (setting the normalisation). In between,
        it acts as an uncertainty on this value.

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

    >>> from mrpy import MRP
    >>> import numpy as np
    >>> m = np.logspace(10,15,500)
    >>> d = MRP(m,14.0,-1.9,0.75,norm=0).dndlog10m()

    Then find the best-fit parameters for the resulting data:

    >>> from mrpy.fitting.fit_curve import get_fit_curve
    >>> res,curve = get_fit_curve(m,d,x0=[14.2,-1.8,0.8,0.5])
    >>> print res.x

    We can also use the ``curve`` object to explore some of the qualities of the fit

    >>> print curve.hessian
    >>> from matplotlib.pyplot import plot
    >>> plot(curve.logm,curve.dndm(log=True))
    >>> plot(curve.logm,np.log(d))
    >>> print curve.stats.mean

    """

    # Input verification
    if len(x0) != 4:
        if len(x0)==3 and sig_rhomean==0.0:
            pass
        else:
            raise ValueError("Need to pass four parameters to fit in x0")

    if sig_rhomean==0  and len(x0)>3:
        print("Warning: x0 was passed with four values, but only requires three")
        x0 = x0[:3]
        if bounds is not None:
            bounds = bounds[:3]

    assert bounds is None or len(bounds)==len(x0), "Bounds must be None or same length as x0"

    # Define the objective function for minimization.
    def model(x0):
        x0 = list(x0)
        if len(x0)==3:
            if sig_rhomean == 0.0:
                x0.append("rhom")

        _curve = lk.CurveLike(np.log10(data_m), data_mf, *x0,
                              sig_data = sig_data, sig_rhomean=sig_rhomean)

        if jac:
            return -_curve.lnL, -_curve.jacobian
        else:
            return -_curve.lnL

    res = opt.minimize(model, x0, bounds=bounds, jac=jac, **minimize_kw)

    xx = list(res.x)
    if len(xx) == 3:
        xx.append('rhom')

    c = lk.CurveLike(np.log10(data_m), data_mf, *xx,
                     sig_data = sig_data, sig_rhomean=sig_rhomean)

    return res, c


def get_fit_expected(data_m, data_mf, x0,
                     bounds=None, jac=True,
                     V0=1, kappa=None,
                     **minimize_kw):
    """
    Generate the expected MLE of MRP parameters, given a mass function from which the samples should be drawn.

    This integrates over the input mass function, so as to naturally weight each mass scale as if fitting an actual
    sample. It assumes that there are no uncertainties on the mass estimates.

    Parameters
    ----------
    data_m : array
        Masses (not log)

    data_mf : array
        Mass function corresponding to m.

    x0 : float, optional
        Initial guess for each of the MRP parameters (hs, alpha, beta, lnA)

    bounds : list of tuples, optional
        If None, don't use bounds. If true, set bounds based on bounds passed.

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

    >>> from mrpy import MRP
    >>> import numpy as np
    >>> m = np.logspace(10,15,500)
    >>> d = MRP(m,14.0,-1.9,0.75,norm=0).dndlog10m()

    Then find the best-fit parameters for the resulting data:

    >>> from mrpy.fitting.fit_curve import get_fit_expected
    >>> res,curve = get_fit_curve(m,d,x0=[14.2,-1.8,0.8,0.5])
    >>> print res.x

    We can also use the ``curve`` object to explore some of the qualities of the fit

    >>> print curve.hessian
    >>> from matplotlib.pyplot import plot
    >>> plot(curve.logm,curve.dndm(log=True))
    >>> plot(curve.logm,np.log(d))
    >>> print curve.stats.mean

    """

    res = opt.minimize(lambda x : -lk.expected_likelihood(x, np.log10(data_m),data_mf,kappa,V0), x0, bounds=bounds, jac=jac, **minimize_kw)
    c = MRP(data_m, res.x[0], res.x[1], res.x[2], norm = res.x[3])

    return res, c
