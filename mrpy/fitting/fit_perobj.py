"""
Routines that implement fits directly to samples of masses (or other values
drawn from a TGGD).

For fits to binned data, see :module:`mrpy.fit_curve`. For the
definition of the likelihood involved in the fits within this module see
:class:`mrpy.likelihoods.PerObjLike`.

This module also provides pre-defined prior functions, specifically, the ``normal_prior``.
"""

from itertools import product
import numpy as np
import scipy.optimize as opt
from mrpy.special import gammainc
from scipy.stats import truncnorm

import mrpy.extra.likelihoods as lk

try:
    import emcee
except ImportError:
    print("Warning: emcee not installed, some routines won't work.")

try:
    import pystan
except:
    print("Warning: pystan not installed, some routines won't work.")


def _retarg(ll, jac, ret_jac):
    if ret_jac:
        return ll, jac
    else:
        return ll


# Define the likelihood function
def _lnl(p, m, nm, mmin, V,  bounds, prior_func=None, prior_kwargs={},debug=0, ret_jac=False):
    ## Note m,nm, mmin are always interpreted as being a list of arrays/scalars

    # Some absolute bounds
    if p[2] < 0 or p[0] < 0 :
        if debug > 0:
            print("OUT OF BOUNDS: ", p)
        return _retarg(-np.inf, np.inf, ret_jac)

    # Enforced bounds
    for i in range(len(p)):
        if not bounds[i][0] <= p[i] <= bounds[i][1]:
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
        _mod = lk.PerObjLikeWeights(weights=nmi, logm=np.log10(mi),
                                    log_mmin=np.log10(mmini),
                                    logHs=p[0], alpha=p[1], beta=p[2],lnA = p[3] + np.log(Vi))
        ll += _mod.lnL
        if ret_jac:
            jac += _mod.jacobian

    if debug > 1:
        print("pars, ll, jac: ", p, ll, jac)

    if np.isnan(ll):
        return _retarg(-np.inf, jac, ret_jac)
    else:
        return _retarg(ll, jac, ret_jac)


def _lnl_with_uncertainty__(p, m_obs, nm, sd_dex, V,  bounds, prior_func=None, prior_kwargs={},debug=0):
    ## Note m,nm, sd_dex, mmin are always interpreted as being a list of arrays/scalars
    ## mtrue here is actually mtrue - mmin.

    theta = p[:5]  #hs, alpha, beta, lnA, log_mmin

    ## Re-sort the m_est parameters into lists like m_obs.
    a = len(theta)
    m_est = []
    for i,mi in enumerate(m_obs):
        b = len(mi)
        m_est += [p[a:a+b]]
        a += b

    # Some absolute bounds
    if theta[2] < 0 or theta[0] < 0:
        if debug > 0:
            print("OUT OF BOUNDS: ", p)
        return -np.inf

    for mesti in m_est:
        if np.any(mesti<0):
            return -np.inf

    # Enforced bounds
    for i in range(len(theta)):
        if not bounds[i][0] <= theta[i] <= bounds[i][1]:
            if debug > 0:
                print("parameter out of bounds: ", theta, bounds)
            return -np.inf

    # theta priors
    if prior_func is None:  # default uniform prior.
        ll = 0
    else:
        ll, _ = prior_func(theta,**prior_kwargs)

    # Likelihood
    for m_esti, m_obsi, sdi, nmi, Vi in zip(m_est,m_obs, sd_dex,nm, V):
        mmini = theta[4]
        m_esti += mmini
        _mod = lk.PerObjLikeWeights(weights=nmi, logm=m_esti,
                                    log_mmin=mmini,
                                    logHs=p[0], alpha=p[1], beta=p[2],lnA = p[3] + np.log(Vi))
        # model likelihood
        ll += _mod.lnL + np.sum(m_esti*np.log(10) + np.log(np.log(10)))  ## Last bit converts to dn/dlog10m

        # uncertainties
        ll -= np.sum(nmi*(m_esti - m_obsi)**2/(2*sdi**2))

    if debug > 1:
        print("pars, ll: ", theta, ll)

    if np.isnan(ll):
        return -np.inf
    else:
        return ll

def _lnl_with_uncertainty(p, m_obs, nm, sd_dex, V,  bounds, prior_func=None, prior_kwargs={},debug=0):
    ## Note m,nm, sd_dex, mmin are always interpreted as being a list of arrays/scalars
    ## mtrue here is actually mtrue - mmin.

    theta = p[:4]  #hs, alpha, beta, log_mmin
    m_est = p[4:] + theta[3]

    # Some absolute bounds
    if theta[2] < 0:
        if debug > 0:
            print("OUT OF BOUNDS: ", p)
        return -np.inf

    if np.any(m_est<theta[3]):
        return -np.inf

    # Enforced bounds
    for i in range(len(theta)):
        if not bounds[i][0] <= theta[i] <= bounds[i][1]:
            if debug > 0:
                print("parameter out of bounds: ", theta, bounds)
            return -np.inf

    # theta priors
    if prior_func is None:  # default uniform prior.
        ll = 0
    else:
        ll, _ = prior_func(theta,**prior_kwargs)

    # Likelihood
    gzx = gammainc((theta[1]+1)/theta[2],np.exp(np.log(10)*(theta[3]-theta[0])*theta[2]))
    raw_lnA = np.log(len(m_obs)) - np.log(10)*theta[0] - np.log(gzx)
    #lnA = raw_lnA - np.log(V)
    #y = m_est - theta[0]
    #x = np.exp(np.log(10)*y*theta[2])

    _mod = lk.PerObjLikeWeights(weights=np.ones_like(m_obs), logm=m_est,
                                log_mmin=theta[3],
                                logHs=p[0], alpha=p[1], beta=p[2], lnA=raw_lnA)
    # model likelihood
    ll += _mod.lnL + np.sum(m_est*np.log(10) + np.log(np.log(10)))  ## Last bit converts to dn/dlog10m

    #ll += np.sum(raw_lnA + np.log(theta[2]) + np.log(np.log(10)) + np.log(10)*theta[0] + np.log(10)*y*(theta[1] + 1) - x) - np.exp(raw_lnA)*10**theta[0]*gzx
    if debug > 0:
        print("raw GGD ll: ", ll, raw_lnA, gzx)
        # # model likelihood
        # ll += _mod.lnL + np.sum(m_esti*np.log(10) + np.log(np.log(10)))  ## Last bit converts to dn/dlog10m

    # uncertainties
    ll -= np.sum((m_est - m_obs)**2/(2*sd_dex**2))
    if debug:
        print("mass mvmt ll: ", -np.sum((m_est - m_obs)**2/(2*sd_dex**2)))
    if debug > 1:
        print("pars, ll: ", theta, ll)

    if np.isnan(ll):
        return -np.inf
    else:
        return ll

def _lnl_with_uncertainty_poly(p, m_obs, nm, sd_dex, MS, V,  bounds, prior_func=None, prior_kwargs={},debug=0):
    ## Note m,nm, sd_dex, mmin are always interpreted as being a list of arrays/scalars

    theta = p[:4]  #hs, alpha, beta, lnA

    polynodes = p[4:]

    # Some absolute bounds
    if theta[2] < 0 or theta[0] < 0:
        if debug > 0:
            print("OUT OF BOUNDS: ", p)
        return -np.inf

    # Enforced bounds
    for i in range(len(theta)):
        if not bounds[i][0] <= theta[i] <= bounds[i][1]:
            if debug > 0:
                print("parameter out of bounds: ", theta, bounds)
            return -np.inf

    # theta priors
    if prior_func is None:  # default uniform prior.
        ll = 0
    else:
        ll, _ = prior_func(theta,**prior_kwargs)

    # Likelihood
    for ms, m_obsi, sdi, nmi, Vi in zip(MS,m_obs, sd_dex,nm, V):
        m_esti = m_obsi + np.dot(polynodes,ms)
        mmini = m_esti.min()
        _mod = lk.PerObjLikeWeights(weights=nmi, logm=m_esti,
                                    log_mmin=mmini,
                                    logHs=p[0], alpha=p[1], beta=p[2],lnA = p[3] + np.log(Vi))
        # model likelihood
        ll += _mod.lnL + np.sum(m_esti*np.log(10) + np.log(10))  ## Last bit converts to dn/dlog10m
        print(ll)
        # uncertainties
        ll -= np.sum(nmi*(m_esti - m_obsi)**2/(2*sdi**2))
        print(np.sum(nmi*(m_esti - m_obsi)**2/(2*sdi**2)))
    if debug > 1:
        print("pars, ll: ", theta, ll)

    if np.isnan(ll):
        return -np.inf
    else:
        return ll


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

class PerObjFit(object):
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

    weight_scale : float, optional
        Mass scaling. Setting this greater than 0 upweights higher valued
        variates, `m`, with a weight of ``m**weight_scale``.

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

    def __init__(self, m, nm=None, mmin=None,sd_dex=None,V=1.0,
                 hs_bounds=(10, 16), alpha_bounds=(-1.99, -1.3),
                 beta_bounds=(0.1, 2.0), lnA_bounds = (-40,-10),
                 n_mnodes=None,n_snodes=None,
                 prior_func=None,prior_kwargs={}):

        self._determine_suite(m, nm, mmin,sd_dex,V)

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
#        self.weight_scale = weight_scale
        self.n_snodes = n_snodes
        self.n_mnodes = n_mnodes


        if n_mnodes is None and n_snodes is not None:
            self.n_mnodes = 1
        elif n_snodes is None and n_mnodes is not None:
            self.n_snodes = 1

        if n_mnodes is not None:
            smax = max([s.max() for s in self.sd_dex])
            self.MS = [np.array([(m-min(self.log_mmin))**i*(smax-s)**j
                            for i,j in product(list(range(self.n_mnodes)),list(range(self.n_snodes)))])
                  for m,s in zip(self.logm,self.sd_dex)]

    def _determine_suite(self, m, nm, mmin,sd_dex,V):
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

            if sd_dex is None:
                self.sd_dex = None
            else:
                assert len(sd_dex) == len(m)
                self.sd_dex = [sd_dex]

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

            if sd_dex  is None:
                self.sd_dex = None
            else:
                assert len(sd_dex) == len(m)
                self.sd_dex = sd_dex

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

        >>> from mrpy.fitting.fit_perobj import PerObjFit
        >>> FitObj = PerObjFit(r)
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
        if self.sd_dex is not None:
            raise NotImplementedError("Running downhill-gradient methods with mass uncertainties is not yet implemented.")

        p0 =[hs0, alpha0, beta0,lnA0]
        bounds = [self.hs_bounds, self.alpha_bounds, self.beta_bounds,self.lnA_bounds]

        self.downhill_res = opt.minimize(_objfunc, p0, args=(self.m, self.nm, self.mmin, self.V, bounds,
                                                             self.prior_func,self.prior_kwargs,
                                                             debug, jac),
                                         bounds=bounds, jac=jac, **minimize_kw)

        self.downhill_obj = [lk.PerObjLikeWeights(logm=np.log10(mi), weights=nmi,
                                               logHs=self.downhill_res.x[0], alpha=self.downhill_res.x[1],
                                               beta=self.downhill_res.x[2], lnA = self.downhill_res.x[3] - np.log(V),
                                               log_mmin=np.log10(mmini)) for mi, nmi, mmini,V in
                             zip(self.m, self.nm, self.mmin,self.V)]

        return self.downhill_res, self.downhill_obj

    # =========================================================================================
    # EMCEE BASED ROUTINES
    # =========================================================================================
    def _get_cuts(self, bound, mu, sigma):
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
                 opt_init=False, opt_kw={}, chainfile="chain.dat", save_latent = True,
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

        >>> from mrpy.fitting.fit_perobj import PerObjFit
        >>> FitObj = PerObjFit(r)
        >>> mcmc_res = FitObj.run_mcmc(nchains=10,warmup=100,iterations=100)
        >>> print mcmc_res.flatchain.mean(axis=0)
        """
        # First, set guess, either by optimization or passed values
        guess = np.array([hs0,alpha0,beta0,lnA0])
        bounds = [self.hs_bounds, self.alpha_bounds, self.beta_bounds,self.lnA_bounds]

        if opt_init:
            if self.sd_dex is not None:
                raise NotImplementedError("Can't yet optimize with uncertainties")

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


        if self.sd_dex is not None:
            if self.n_mnodes is not None:
                guess_coeffs = np.zeros((self.n_snodes*self.n_mnodes))
                guess_coeffs[0] += 3.5 * max([s.max() for s in self.sd_dex])
                initial_mest = self._get_initial_ball(guess_coeffs, [(g-1,g+1) for g in guess_coeffs],nchains)

            else:
                if logm0 is None:
                    initial_mest = self._get_initial_ball(np.array([np.log10(m) for m in self.m]).flatten(),None,nchains)
                else:
                    initial_mest = self._get_initial_ball(logm0,None,nchains)

            initial = np.concatenate((initial, initial_mest),axis=1)

        if self.sd_dex is None:
            self.mcmc_res = emcee.EnsembleSampler(nchains, initial.shape[1], _lnl,
                                                  args=(self.m, self.nm, self.mmin, self.V, bounds,
                                                        self.prior_func, self.prior_kwargs,
                                                        debug, False), **kwargs)
        elif self.n_mnodes is None:
            self.mcmc_res = emcee.EnsembleSampler(nchains, initial.shape[1], _lnl_with_uncertainty,
                                                  args=(self.logm,
                                                        self.nm, self.sd_dex,self.V, bounds,
                                                        self.prior_func, self.prior_kwargs,
                                                        debug), **kwargs)
        else:
            self.mcmc_res = emcee.EnsembleSampler(nchains, initial.shape[1], _lnl_with_uncertainty_poly,
                                                  args=(self.logm,
                                                        self.nm, self.sd_dex,self.MS,self.V, bounds,
                                                        self.prior_func, self.prior_kwargs,
                                                        debug), **kwargs)
        if warmup:
            initial, lnprob, rstate = self.mcmc_res.run_mcmc(initial, warmup, storechain=False)
            self.mcmc_res.reset()

        if self.sd_dex is None or self.n_mnodes is not None:
            self.mcmc_res.run_mcmc(initial, iterations)
        else:
            f = open(chainfile, "w")
            f.close()

            if save_latent:
                f = open(chainfile.replace(".","_latent."), "w")
                f.close()

            for result in self.mcmc_res.sample(initial, iterations=iterations, storechain=False):
                position = result[0]

                with open(chainfile, "a") as f:
                    np.savetxt(f,position[:,:4])

                if save_latent:
                    with open(chainfile.replace(".","_latent."),"a") as f:
                        np.save(f,position[:,4:])
#                    for k in range(position.shape[0]):
#                        f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))

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
        if self.sd_dex is None:
            return _lnl(p, self.m, self.nm, self.mmin, self.V,bounds, self.prior_func,self.prior_kwargs,debug, ret_jac)
        elif self.n_mnodes is None:
            return _lnl_with_uncertainty(p, self.logm, self.nm, self.sd_dex, self.V,
                                         bounds, self.prior_func,self.prior_kwargs,debug)
        else:
            return _lnl_with_uncertainty_poly(p, self.logm, self.nm, self.sd_dex, self.MS, self.V,
                                              bounds, self.prior_func,self.prior_kwargs,debug)



