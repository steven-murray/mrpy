"""
Routines that implement fits directly to samples of masses (or other values
drawn from a TGGD).

For fits to binned data, see :module:`mrpy.fit_curve`. For the
definition of the likelihood involved in the fits within this module see
:class:`mrpy.likelihoods.PerObjLike`.

This module also provides pre-defined prior functions, specifically, the ``normal_prior``.
"""
import pickle
import hashlib
from os import path
import os
import numpy as np
import scipy.optimize as opt
import likelihoods as lk
#from likelihoods import PerObjLikeWeights
from scipy.stats import truncnorm

try:
    import emcee
except ImportError:
    print "Warning: emcee not installed, some routines won't work."

try:
    import pystan
except:
    print "Warning: pystan not installed, some routines won't work."


def _retarg(ll, jac, ret_jac):
    if ret_jac:
        return ll, jac
    else:
        return ll


# Define the likelihood function
def _lnl(p, m, nm, mmin, s, hs_bounds, alpha_bounds,
         beta_bounds, prior_func=None, prior_kwargs={},debug=0, ret_jac=False):
    ## Note m,nm, mmin are always interpreted as being a list of arrays/scalars

    # Some absolute bounds
    if p[2] < 0 or p[0] < 0:
        if debug > 0:
            print "OUT OF BOUNDS: ", p
        return _retarg(-np.inf, np.inf, ret_jac)

    # Enforced bounds
    if not hs_bounds[0] <= p[0] <= hs_bounds[1]:
        if debug > 0:
            print "logHs out of bounds: ", p
        return _retarg(-np.inf, np.inf, ret_jac)
    if not alpha_bounds[0] <= p[1] <= alpha_bounds[1]:
        if debug:
            print "alpha out of bounds: ", p
        return _retarg(-np.inf, np.inf, ret_jac)
    if not beta_bounds[0] <= p[2] <= beta_bounds[1]:
        if debug:
            print "beta out of bounds: ", p
        return _retarg(-np.inf, np.inf, ret_jac)

    # Priors
    if prior_func is None:  # default uniform prior.
        ll, jac = 0, np.zeros(3)
    else:
        ll, jac = prior_func(p,**prior_kwargs)

    # Likelihood
    for mi, nmi, mmini in zip(m, nm, mmin):
        _mod = lk.PerObjLikeWeights(weights=nmi, scale=s, logm=np.log10(mi),
                                 log_mmin=np.log10(mmini),
                                 logHs=p[0], alpha=p[1], beta=p[2])
        ll += _mod.lnL

        if ret_jac:
            jac += _mod.jacobian

    if debug > 1:
        print "pars, ll, jac: ", p, ll, jac

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

    def __init__(self, m, nm=None, mmin=None, weight_scale=0,
                 hs_bounds=(10, 16), alpha_bounds=(-1.99, -1.3),
                 beta_bounds=(0.1, 2.0), prior_func=None,prior_kwargs={}):

        self._determine_suite(m, nm, mmin)

        # Make sure all masses are above mmin
        for i, (m, mmin) in enumerate(zip(self.m, self.mmin)):
            self.m[i] = m[m >= mmin]
            self.nm[i] = self.nm[i][m >= mmin]

        self.hs_bounds = hs_bounds
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.prior_func = prior_func
        self.prior_kwargs = prior_kwargs
        self.weight_scale = weight_scale

    def _determine_suite(self, m, nm, mmin):
        ## Determine whether there is a suite of simulations.
        if np.isscalar(m[0]):
            self.m = [m]

            if nm is None:
                self.nm = [np.ones_like(m)]
            else:
                self.nm = [nm]

            if mmin is None:
                self.mmin = [m.min()]
            else:
                self.mmin = [mmin]
        else:
            self.m = m

            if nm is None:
                self.nm = [np.ones_like(x) for x in m]
            else:
                self.nm = nm

            if mmin is None:
                self.mmin = [x.min() for x in m]
            else:
                self.mmin = mmin

    def run_downhill(self, hs0=14.5, alpha0=-1.9, beta0=0.8, debug=0, jac=True, **minimize_kw):
        """
        Downhill-gradient optimization.

        Parameters
        ----------
        hs0, alpha0, beta0: float, optional
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

        >>> from mrpy.stats import TGGD
        >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)

        Then find the best-fit parameters for the resulting data:

        >>> from mrpy.fit_perobj import PerObjFit
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
        p0 = [hs0, alpha0, beta0]
        bounds = [self.hs_bounds, self.alpha_bounds, self.beta_bounds]
        self.downhill_res = opt.minimize(_objfunc, p0, args=(self.m, self.nm, self.mmin, self.weight_scale, (0, np.inf),
                                                         (-np.inf, np.inf), (0, np.inf), self.prior_func,self.prior_kwargs,
                                                         debug, jac),
                                     bounds=bounds, jac=jac, **minimize_kw)

        self.downhill_obj = [lk.PerObjLikeWeights(scale=self.weight_scale, logm=np.log10(mi), weights=nmi,
                                               logHs=self.downhill_res.x[0], alpha=self.downhill_res.x[1],
                                               beta=self.downhill_res.x[2],
                                               log_mmin=np.log10(mmini)) for mi, nmi, mmini in
                             zip(self.m, self.nm, self.mmin)]

        return self.downhill_res, self.downhill_obj

    # =========================================================================================
    # EMCEE BASED ROUTINES
    # =========================================================================================
    def _get_cuts(self, bound, mu, sigma):
        return (bound - mu)/sigma

    def _get_initial_ball(self, guess, chains):
        s = 0.05
        a = np.array([self.hs_bounds[0], self.alpha_bounds[0], self.beta_bounds[0]])
        b = np.array([self.hs_bounds[1], self.alpha_bounds[1], self.beta_bounds[1]])

        a = self._get_cuts(a, guess, np.abs(s*guess))
        b = self._get_cuts(b, guess, np.abs(s*guess))
        stacked_val = np.empty((chains, len(guess)))
        for i, (g, A, B) in enumerate(zip(guess, a, b)):
            stacked_val[:, i] = truncnorm(A, B, loc=g, scale=np.abs(s*g)).rvs(chains)

        return stacked_val

    def run_mcmc(self, nchains=50, warmup=1000, iterations=1000,
                 hs0=14.5, alpha0=-1.9, beta0=0.8, debug=0,
                 opt_init=False, opt_kw={}, **kwargs):
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

        >>> from mrpy.stats import TGGD
        >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)

        Then find the best-fit parameters for the resulting data:

        >>> from mrpy.fit_perobj import PerObjFit
        >>> FitObj = PerObjFit(r)
        >>> mcmc_res = FitObj.run_mcmc(nchains=10,warmup=100,iterations=100)
        >>> print mcmc_res.flatchain.mean(axis=0)
        """
        # First, set guess, either by optimization or passed values
        guess = np.array([hs0, alpha0, beta0])
        if opt_init:
            if not hasattr(self, "downhill_res"):
                self.run_downhill(hs0, alpha0, beta0, debug, **opt_kw)
            res = self.downhill_res

            if res.success:
                guess = res.x
                if debug:
                    print "Optimization result (used as guess): ", guess
            else:
                print "WARNING: Optimization failed. Falling back on given initial parameters."

        initial = self._get_initial_ball(guess, nchains)

        self.mcmc_res = emcee.EnsembleSampler(nchains, initial.shape[1], _lnl,
                                              args=(self.m, self.nm, self.mmin, self.weight_scale, self.hs_bounds,
                                                    self.alpha_bounds, self.beta_bounds, self.prior_func, self.prior_kwargs,
                                                    debug, False), **kwargs)

        if warmup:
            initial, lnprob, rstate = self.mcmc_res.run_mcmc(initial, warmup, storechain=False)
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
        return _lnl(p, self.m, self.nm, self.mmin, self.weight_scale, self.hs_bounds,
                    self.alpha_bounds, self.beta_bounds, self.prior_func,
                    debug, ret_jac)

# =========================================================================================
# STAN ROUTINES
# =========================================================================================
_functions_block = """
functions {
    /**
    * gammainc() is the upper incomplete gamma function (not regularized)
    * @param real a, shape parameter
    * @param real x, position > 0
    * @returns the non-regularized incomplete gamma
    * NOTES: uses a recursion relation to calculate values for negative a.
    */
    real gammainc(real a, real x){
      int n;
      real ap1;
      real ssum;

      if(a>=0) return gamma_q(a,x) * tgamma(a);

      ap1 <- a+1;

      //Get floor(-a)
      n<-0;
      while(n<-a){
        n <- n+1;
      }

      //Get summed part
      {
        vector[n] sums;
        for(i in 0:n-1) sums[i+1] <- pow(x,i)/tgamma(ap1+i);
        ssum <- sum(sums);
      }
      return tgamma(a)*(gamma_q(a+n,x)-pow(x,a)*exp(-x)*ssum);
    }

    /**
    * truncated_logGGD_log gives the log PDF of a variate whose exponential has a
    *    lower-truncated generalised gamma distribution.
    * @param vector y, variate (should be log10(m/Hs) in the usual TGGD)
    * @param real ymin, truncation in y-space.
    * @param real alpha, power-law slope
    * @param real beta, cut-off parameter
    */
    real truncated_logGGD_log(vector y, real ymin, real alpha, real beta){
        vector[num_elements(y)] x;
        real xmin;
        real z;

        z <- (alpha+1)/beta;
        x <- exp(log10()*y*beta);
        xmin <- exp(log10()*ymin*beta);
        return sum(log(beta) + log(log10()) + log10()*y*(alpha+1) - x - log(gammainc(z,xmin)));
    }
}
"""

_with_errors_data = """
data {
    int<lower=0> N;                // number of halos
    vector<lower=0>[N] log_m_meas; // measured halo masses
    vector<lower=0>[N] sd_dex;     // uncertainty in halo masses (dex)

    // CONTROLS FOR PARAMETER BOUNDS
    real<lower=0> hs_min;             // Lower bound of logHs
    real<lower=0,upper=20> hs_max;    // Upper bound of logHs
    real<lower=-2,upper=0> alpha_min; // Lower bound of alpha
    real<lower=-2,upper=0> alpha_max; // Upper bound of alpha
    real<lower=0> beta_min;           // Lower bound of beta
    real<lower=0> beta_max;           // Upper bound of beta
    real<lower=0> mmin_min;           // Lower bound of log_mmin
    real<lower=0> mmin_max;           // Upper bound of log_mmin
    real<lower=0,upper=20>mtrue_max;  // Upper bound of true masses
}
"""

_simple_data = """
data {
    int<lower=0> N;                // number of halos
    vector<lower=0>[N] log_m;      // measured halo masses

    // CONTROLS FOR PARAMETER BOUNDS
    real<lower=0> hs_min;             // Lower bound of logHs
    real<lower=0,upper=20> hs_max;    // Upper bound of logHs
    real<lower=-2,upper=0> alpha_min; // Lower bound of alpha
    real<lower=-2,upper=0> alpha_max; // Upper bound of alpha
    real<lower=0> beta_min;           // Lower bound of beta
    real<lower=0> beta_max;           // Upper bound of beta
}
transformed data {
    real<lower=0> log_mmin;
    log_mmin <- min(log_m);
}
"""

_with_errors_params = """
parameters {
    real<lower=hs_min,upper=hs_max> logHs;               // Characteristic halo mass
    real<lower=alpha_min,upper=alpha_max> alpha;         // Power-law slope
    real<lower=beta_min,upper=beta_max> beta;            // Cut-off parameter
    real<lower=mmin_min,upper=mmin_max> log_mmin;        // Truncation mass
    vector<lower=log_mmin,upper=mtrue_max>[N] log_mtrue; // True mass estimates
}
"""

_simple_params = """
parameters {
    real<lower=hs_min,upper=hs_max> logHs;               // Characteristic halo mass
    real<lower=alpha_min,upper=alpha_max> alpha;         // Power-law slope
    real<lower=beta_min,upper=beta_max> beta;            // Cut-off parameter
}
"""

_with_errors_model = """
model {
    vector[N] y;
    real ymin;
    y <- log_mtrue-logHs;
    ymin <- log_mmin-logHs;

    y ~ truncated_logGGD(ymin, alpha, beta);
    log_mtrue ~ normal(log_m_meas,sd_dex);
}
"""

_simple_model = """
model {
    vector[N] y;
    real ymin;
    y <- log_m-logHs;
    ymin <- log_mmin-logHs;

    y ~ truncated_logGGD(ymin, alpha, beta);
}
"""


def _create_model(per_object_errors=False):
    if per_object_errors:
        return _functions_block + _with_errors_data + _with_errors_params + _with_errors_model
    else:
        return _functions_block + _simple_data + _simple_params + _simple_model


def _write_model(fname, per_object_errors=False):
    s = _create_model(per_object_errors)
    with open(fname, "w") as f:
        f.write(s)


def _compile_model(per_object_errors=False):
    return _stan_cache(model_name="MRP", model_code=_create_model(per_object_errors))


def _stan_cache(model_code, model_name=None):
    code_hash = hashlib.md5(model_code.encode('ascii')).hexdigest()

    # Find the mrpy project dir
    dir = path.join(path.expanduser("~"), '.mrpy')
    if not path.exists(dir):
        os.makedirs(dir)

    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)

    try:
        sm = pickle.load(open(path.join(dir, cache_fn), 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code, model_name=model_name)
        with open(path.join(dir, cache_fn), 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")

    return sm


def fit_perobj_stan(m, sd_dex=None, warmup=None, iter=1000,
                    hs_bounds=(12, 16), alpha_bounds=(-1.99, -1.3),
                    beta_bounds=(0.3, 2.0), mmin_bounds=None, opt=False,
                    mtrue_max=16.0, model=None, **kwargs):
    """
    Fit the MRP to individual halo masses using the Stan programming language.

    This method has less options than its `emcee` counterpart :func:`fit_perobj_emcee`,
    but importantly can handle arbitrary per-object errors on the masses. For example,
    at this stage, the priors on each parameter are hard-coded to be uniform.

    Parameters
    ----------
    m : array_like
        The masses (or variates).

    sd_dex : array_like
        Either a scalar giving the same lognormal uncertainty for each variate, or
        a vector of the same length as `m`, giving arbitrary errors for each mass.

    iter : int
        The number of iterations to run.

    warmup : int
        The number of (discarded) warmup iterations to run (default ``iter/2``).

    hs_bounds, alpha_bounds, beta_bounds : 2-tuples
        Sets the boundaries of the parameters.

    mmin_bounds : 2-tuple
        The bounds for the mmin parameter, if errors are given on the masses.

    opt : bool
        Whether to run downhill optimization instead of MCMC. This does not work
        when errors are provided on the masses.

    mtrue_max : float
        The maximum value that an estimated mass can have. Only used if errors
        are given for masses.

    kwargs
        Passed to :func:`StanModel.sampling`. Please look at the pystan docs.

    Returns
    -------
    fit : :class:`pystan.StanFit4Model` object
        An object containing the stored chains and other methods to analyse
        the results.
        
    """
    if sd_dex is None:
        stan_data = {"N": len(m),
                     "log_m": np.log10(m),
                     "hs_min": hs_bounds[0],
                     "hs_max": hs_bounds[1],
                     "alpha_min": alpha_bounds[0],
                     "alpha_max": alpha_bounds[1],
                     "beta_min": beta_bounds[0],
                     "beta_max": beta_bounds[1]}
    else:
        if np.isscalar(sd_dex):
            sd_dex = np.repeat(sd_dex, len(m))

        if mmin_bounds is None:
            mmin_bounds = (np.log10(m.min()) - 1, np.log10(m.min()) + 1)

        stan_data = {"N": len(m),
                     "log_m_meas": np.log10(m),
                     "sd_dex": sd_dex,
                     "hs_min": hs_bounds[0],
                     "hs_max": hs_bounds[1],
                     "alpha_min": alpha_bounds[0],
                     "alpha_max": alpha_bounds[1],
                     "beta_min": beta_bounds[0],
                     "beta_max": beta_bounds[1],
                     "mmin_min": mmin_bounds[0],
                     "mmin_max": mmin_bounds[1],
                     "mtrue_max": mtrue_max}

    if model is None:
        model = _compile_model(False if sd_dex is None else True)

    warmup = warmup or iter/2

    if opt:
        fit = model.optimizing(data=stan_data)
    else:
        fit = model.sampling(stan_data, iter=iter, warmup=warmup, **kwargs)

    return fit
