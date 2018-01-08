"""
Routines that implement fits directly to samples of masses (or other values
drawn from a TGGD).

For fits to binned data, see :module:`mrpy.fit_curve`. For the
definition of the likelihood involved in the fits within this module see
:class:`mrpy.likelihoods.PerObjLike`.

This module also provides pre-defined prior functions, specifically, the ``normal_prior``.
"""

import numpy as np
import scipy.optimize as opt
from scipy.stats import truncnorm

import mrpy.extra.likelihoods as lk

try:
    import emcee
except ImportError:
    print("Warning: emcee not installed, some routines won't work.")


def _retarg(ll, jac, ret_jac):
    if ret_jac:
        return ll, jac
    else:
        return ll


# Define the likelihood function
def _lnl(p, m, nm, mmin, V,  bounds, prior_func=None, prior_kwargs=None,debug=0, ret_jac=False):
    ## Note m,nm, mmin are always interpreted as being a list of arrays/scalars

    if prior_kwargs is None:
        prior_kwargs = {}

    # Some absolute bounds
    if p[2] < 0 or p[0] < 0 :
        if debug > 0:
            print("OUT OF BOUNDS: ", p)
        return _retarg(-np.inf, np.inf, ret_jac)

    # Enforced bounds
    for i,pp in enumerate(p):
        if not bounds[i][0] <= pp <= bounds[i][1]:
            if debug > 0:
                print("parameter out of bounds: ", p, bounds)
            return _retarg(-np.inf, np.inf, ret_jac)

    # Priors
    if prior_func is None:  # default uniform prior.
        ll, jac = 0, np.zeros_like(p)
    else:
        ll, jac = prior_func(p,**prior_kwargs)

    # Likelihood
    for mi, nmi, mmini,Vi in zip(m, nm, mmin,V):
        _mod = lk.SampleLikeWeights(weights=nmi, logm=np.log10(mi),
                                    log_mmin=np.log10(mmini),
                                    logHs=p[0], alpha=p[1], beta=p[2], lnA = p[3] + np.log(Vi))
        ll += _mod.lnL
        if ret_jac:
            jac += _mod.jacobian

    if debug > 1:
        print("pars, ll, jac: ", p, ll, jac)

    if np.isnan(ll):
        return _retarg(-np.inf, jac, ret_jac)
    else:
        return _retarg(ll, jac, ret_jac)

def _objfunc(*args):
    out = _lnl(*args)
    if np.isscalar(out):
        return -out
    else:
        return -out[0], -out[1]

def normal_prior(p,mean,sd):
    """
    A normal prior on each parameter.

    Parameters
    ----------
    p : list
        Values of the parameters

    mean : list
        Values of the prior mean

    sd : list
        Values of the prior standard deviations. Set to inf
        if uniform prior required on a single parameter.

    Returns
    -------
    ll :
        The likelihood of the parameters given the priors

    jac :
        The jacobian of the likelihood.

    """
    ll = 0
    jac = []

    for v,mu,s in zip(p,mean,sd):
        ll += -((v-mu)/(2*s))**2
        jac.append(-(v-mu)/s)

    return ll, jac


class SimFit(object):
    """
    Per-object fits.

    Parameters
    ----------
    m : array or list of arrays
        Masses. Either an array or a list of arrays, each of which is a sample *to be
        analysed simultaneously*. In the latter case the samples should have the same
        underlying distribution, but may have differing truncation scales.

    nm : array, optional
        Specifies the number of occurrences of each variate in `m` (which should then
        ideally be unique). If not passed, each variate is assumed to occur once. This
        is useful for speeding up fits on quantized simulations. If `m` is a list of
        arrays, this should be also.

    mmin : array_like, optional
        The truncation mass of the sample. By default takes the lowest value of `m`.
        If `m` is a list of arrays, this should be a list.

    V : array, optional
        The volume of each subsample

    hs_bounds, alpha_bounds, beta_bounds : 2-tuple
        2-tuples specifying minimum and maximum values for each bound.

    prior_func : function, optional
        A function to calculate the likelihood arising from priors on the parameters.
        By default, uniform priors are assumed, which add nothing to the likelihood.
        The first parameter taken by the function should be the vector of parameter
        values, ``[logHs,alpha,beta]``, after which arbitrary values are passed via
        `prior_kwargs`. It should return a tuple, with the first value
        being a float and constituting the likelihood arising from the prior, and the second
        being a 3-vector constituting the modification to the jacobian. This can be zero
        if no jacobian is desired.

    prior_kwargs : dict
        Arguments sent to the `prior_func`.

    Notes
    -----
    Use as stringent bounds as possible, since the algorithm explores the
    edges, which can induce numerical error if values far from the solution are chosen.

    The setting of `weight_scale` can be tricky. It is sometimes necessary to set it higher than 0
    to achieve reasonable precision on `beta`, due to the severe relative lack of high-value
    variates. Nevertheless, doing so in general decreases the reliability of the fit. Simple
    tests show that in typical cases, about 10 times as many variates are required for
    a ``weight_scale=1`` fit to achieve the same accuracy on parameters.
    """

    def __init__(self, m, nm=None, mmin=None,V=1.0,
                 hs_bounds=(10, 16), alpha_bounds=(-1.99, -1.3),
                 beta_bounds=(0.1, 2.0), lnA_bounds = (-40,-10),
                 prior_func=None,prior_kwargs=None):

        if prior_kwargs is None:
            prior_kwargs = {}

        self._determine_suite(m, nm, mmin,V)

        # Make sure all masses are above mmin
        for i, (m, mmin) in enumerate(zip(self.m, self.mmin)):
            self.m[i] = m[m >= mmin]
            self.nm[i] = self.nm[i][m >= mmin]

        self.hs_bounds = hs_bounds
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.lnA_bounds = lnA_bounds
        self.prior_func = prior_func
        self.prior_kwargs = prior_kwargs


    def _determine_suite(self, m, nm, mmin,V):
        ## Determine whether there is a suite of simulations.
        if np.isscalar(m[0]):
            self.m = [m]
            self.logm = [np.log10(m)]

            if nm is None:
                self.nm = [np.ones_like(m)]
            else:
                self.nm = [nm]

            if mmin is None:
                self.mmin = [m.min()]
            else:
                self.mmin = [mmin]

            self.log_mmin = [np.log10(self.mmin[0])]

            self.V = np.array([V]).flatten()

        else:
            self.m = m
            self.logm = [np.log10(x) for x in self.m ]

            if nm is None:
                self.nm = [np.ones_like(x) for x in m]
            else:
                self.nm = nm

            if mmin is None:
                self.mmin = [x.min() for x in m]
            else:
                self.mmin = mmin

            self.log_mmin = [np.log10(x.min()) for x in self.mmin]

            if np.isscalar(V):
                print("WARNING: V is a scalar, but there are multiple datasets")
                self.V = [V]*len(m)
            else:
                self.V = V

    def run_downhill(self, hs0=14.5, alpha0=-1.9, beta0=0.8, lnA0=-40.0,
                     debug=0, jac=True, **minimize_kw):
        """
        Downhill-gradient optimization.

        Parameters
        ----------
        hs0, alpha0, beta0, lnA0: float, optional
            Initial guess for each of the MRP parameters.

        debug : int, optional
            Set the level of info printed out throughout the function. Highest current
            level that is useful is 2.

        jac : bool, optional
            Whether to use analytic jacobian (usually a good idea)

        minimize_kw : dict
            Any other parameters to :func:`scipy.optimize.minimize`.

        Returns
        -------
        downhill_res : `OptimizeResult`
            The optimization result represented as a ``OptimizeResult`` object
            (see scipy documentation).
            Important attributes are: ``x`` the solution array, ``success`` a Boolean flag
            indicating if the optimizer exited successfully and ``message`` which describes
            the cause of the termination.

            The parameters are ordered by `logHs`, `alpha`, `beta`, `[lnA]`.

        downhill_obj : :class:`mrpy.likelihoods.PerObjLikeWeights` object
            An object containing the solution parameters and methods to access
            relevant quantities, such as the mass function, or jacobian and
            hessian at the solution.

        Examples
        --------
        The most obvious example is to generate a sample of variates from given parameters:

        >>> from mrpy.base.stats import TGGD
        >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)

        Then find the best-fit parameters for the resulting data:

        >>> from mrpy.fitting.fit_sample import SampleFit
        >>> FitObj = SampleFit(r)
        >>> res,obj = FitObj.run_downhill()
        >>> print res.x

        We can also use the ``obj`` object to explore some of the qualities of the fit

        >>> print obj.hessian
        >>> print obj.cov

        The results are also stored in the class as `downhill_obj` and `downhill_res`.

        >>> from matplotlib.pyplot import plot
        >>> plot(FitObj.downhill_obj.logm,FitObj.downhill_obj.dndm(log=True))
        >>> print obj.stats.mean, r.mean()
        """
        p0 =[hs0, alpha0, beta0,lnA0]
        bounds = [self.hs_bounds, self.alpha_bounds, self.beta_bounds,self.lnA_bounds]

        self.downhill_res = opt.minimize(_objfunc, p0, args=(self.m, self.nm, self.mmin, self.V, bounds,
                                                             self.prior_func,self.prior_kwargs,
                                                             debug, jac),
                                         bounds=bounds, jac=jac, **minimize_kw)

        self.downhill_obj = [lk.SampleLikeWeights(logm=np.log10(mi), weights=nmi,
                                                  logHs=self.downhill_res.x[0], alpha=self.downhill_res.x[1],
                                                  beta=self.downhill_res.x[2], lnA = self.downhill_res.x[3] - np.log(V),
                                                  log_mmin=np.log10(mmini)) for mi, nmi, mmini,V in
                             zip(self.m, self.nm, self.mmin,self.V)]

        return self.downhill_res, self.downhill_obj

    # =========================================================================================
    # EMCEE BASED ROUTINES
    # =========================================================================================
    @staticmethod
    def _get_cuts(bound, mu, sigma):
        return (bound - mu)/sigma

    def _get_initial_ball(self, guess, bounds, chains):
        s = 0.05
        if bounds is not None:
            a = np.array([c[0] for c in bounds])
            b = np.array([c[1] for c in bounds])

            aa = self._get_cuts(a, guess, np.abs(s*(b-a)))
            bb = self._get_cuts(b, guess, np.abs(s*(b-a)))
        else:
            aa = [-np.inf]*len(guess)
            bb = [np.inf]*len(guess)

        stacked_val = np.empty((chains, len(guess)))

        for i, (g, A, B) in enumerate(zip(guess, aa, bb)):
            stacked_val[:, i] = truncnorm(A, B, loc=g, scale=np.abs(s*(b[i]-a[i]))).rvs(chains)

        return stacked_val

    def run_mcmc(self, nchains=50, warmup=1000, iterations=1000,
                 hs0=14.5, alpha0=-1.9, beta0=0.8, lnA0=-26.0, logm0 = None, debug=0,
                 opt_init=False, opt_kw=None, chainfile="chain.dat", save_latent = True,
                 **kwargs):
        """
        Per-object MCMC fit for masses `m`, using the `emcee` package.

        This creates an :class:`emcee.EnsembleSampler` object with a correct
        model, and runs warmup and specified iterations. The entire :class:`emcee.EnsembleSampler`
        object is returned, and stored in the instance of this class as :attr:`~mcmc_res`. This affords
        greater flexibility, with the ability to run no warmup and 0 iterations, and run the iterations
        oneself.


        Parameters
        ----------
        nchains : int, optional
            Number of chains to use in the AIES MCMC algorithm

        warmup : int, optional
            Number (discarded) warmup iterations.

        iterations : int, optional
            Number of iterations to keep in the chain.

        hs0, alpha0, beta0: float, optional
            Initial guess for each of the MRP parameters.

        debug : int, optional
            Set the level of info printed out throughout the function. Highest current
            level that is useful is 2.

        opt_init : bool, optional
            Whether to run a downhill optimization routine to get the best
            starting point for the MCMC.

        opt_kw : dict, optional
            Any arguments to pass to the downhill run.

        kwargs :
            Any other parameters to :class:`emcee.EnsembleSampler`.

        Returns
        -------
        mcmc_res : :class:`emcee.EnsembleSampler` object
            This object contains the stored chains, and other attributes.

        Examples
        --------
        The most obvious example is to generate a sample of variates from given parameters:

        >>> from mrpy.base.stats import TGGD
        >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)

        Then find the best-fit parameters for the resulting data:

        >>> from mrpy.fitting.fit_sample import SampleFit
        >>> FitObj = SampleFit(r)
        >>> mcmc_res = FitObj.run_mcmc(nchains=10,warmup=100,iterations=100)
        >>> print mcmc_res.flatchain.mean(axis=0)
        """
        if opt_kw is None:
            opt_kw = {}

        # First, set guess, either by optimization or passed values
        guess = np.array([hs0,alpha0,beta0,lnA0])
        bounds = [self.hs_bounds, self.alpha_bounds, self.beta_bounds,self.lnA_bounds]

        if opt_init:
            if not hasattr(self, "downhill_res"):
                self.run_downhill(hs0, alpha0, beta0, lnA0, debug, **opt_kw)
            res = self.downhill_res

            if res.success:
                guess = res.x
                if debug:
                    print("Optimization result (used as guess): ", guess)
            else:
                print("WARNING: Optimization failed. Falling back on given initial parameters.")

        initial = self._get_initial_ball(guess, bounds, nchains)

        self.mcmc_res = emcee.EnsembleSampler(nchains, initial.shape[1], _lnl,
                                              args=(self.m, self.nm, self.mmin, self.V, bounds,
                                                    self.prior_func, self.prior_kwargs,
                                                    debug, False), **kwargs)
        if warmup:
            initial, _, rstate = self.mcmc_res.run_mcmc(initial, warmup, storechain=False)
            self.mcmc_res.reset()


        self.mcmc_res.run_mcmc(initial, iterations)

        return self.mcmc_res

    def lnL(self, p, ret_jac=False, debug=0):
        """
        Return the log-likelihood of the current model at the parameters `p`.

        Parameters
        ----------
        p : array
            The values of the parameters, ``[logHs, alpha, beta]``.

        ret_jac : bool
            Whether to return the jacobian as the second arg.

        debug : int, optional
            Set the level of info printed out throughout the function. Highest current
            level that is useful is 2.

        Returns
        -------
        ll : float
            The log-likelihood at the parameters. This is exactly the same as used in the
            fitting process.

        jac : length-3 array, optional
            Returned only if `ret_jac` is `True`. The jacobian at the current parameter vector.

        """
        bounds = [self.hs_bounds, self.alpha_bounds, self.beta_bounds,self.lnA_bounds]
        return _lnl(p, self.m, self.nm, self.mmin, self.V,bounds, self.prior_func,self.prior_kwargs,debug, ret_jac)




