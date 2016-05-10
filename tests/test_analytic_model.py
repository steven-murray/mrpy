"""
Module for testing the analytic/ideal model.

Tests basically pit numerical derivatives against analytic ones.
"""
import inspect
import os
import sys
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
sys.path.insert(0, LOCATION)

from mrpy.analytic_model import IdealAnalytic
from mrpy.likelihoods import PerObjLike
from mrpy._utils import numerical_hess, numerical_jac
import numpy as np
from scipy.integrate import simps

## Set some default parameters to use throughout
alpha = -1.8
beta = 0.7
logHs = 14.0
mmin = 11.0
lnA = 0
scale = 0

# A series of trial parameters
trials = [(logHs, alpha, beta, lnA, logHs, alpha, beta,lnA),#, scale),
          #(logHs, alpha, beta, logHs, alpha, beta),#, 1.0),
          (logHs, alpha, beta, lnA, logHs*1.1, alpha, beta,lnA),#, scale),
          (logHs, alpha, beta, lnA, logHs, alpha*1.05, beta,lnA),#, scale),
          (logHs, alpha, beta, lnA, logHs, alpha, beta*1.1, lnA),
          (logHs, alpha, beta, lnA, logHs, alpha, beta*1.1, lnA+0.5)]# scale)]


def numerical_F(alpha, beta, logHs, lnA, alphad=None, betad=None, logHsd=None,lnAd=None):
    alphad = alphad or alpha
    betad = betad or beta
    logHsd = logHsd or logHs
    lnAd = lnAd or lnA

    m = np.logspace(mmin, 18, 1000)
    mrp = PerObjLike(logm=np.log10(m), logHs=logHs, alpha=alpha, beta=beta, lnA=lnA)
    mrpd = PerObjLike(logm=np.log10(m), logHs=logHsd, alpha=alphad, beta=betad,lnA = lnAd)
    integ = mrpd._g*mrp._lng
    return simps(integ, m) - mrp._q_


def F_abs_ratio(logHs, alpha, beta,lnA, logHsd, alphad, betad,lnAd):
    ia = IdealAnalytic(log_mmin=mmin, logHs=logHs, alpha=alpha, beta=beta, lnA=lnA,logHsd=logHsd,
                       alphad=alphad, betad=betad,lnAd=lnAd)
    num = numerical_F(alpha, beta, logHs, alphad, betad, logHsd)
    print ia._F, num
    assert np.isclose(ia._F,num, rtol=1e-2)


def test_F_abs():
    """
    Test if F, defined as a numerical integral, is the same as the analytic case.
    """
    for h, a, b, lnA, hd, ad, bd,lnAd in trials:
        yield F_abs_ratio, h, a, b, lnA,hd, ad, bd,lnAd


# --------- HELPER FUNCTIONS ----------------------------------------------------------
def ideal_numerical_jh(q, dx=1e-4, hess=False, **kwargs):
    def func(**kwargs):
        return getattr(IdealAnalytic(**kwargs), q)

    if hess:
        return numerical_hess(func, ["logHs", "alpha", 'beta','lnA'], dx, **kwargs)
    else:
        return numerical_jac(func, ["logHs", "alpha", 'beta','lnA'], dx, **kwargs)


def getq_jac(q, **kwargs):
    ia = IdealAnalytic(**kwargs)
    return np.array([getattr(ia, q + "_%s"%x) for x in 'hab'])


def getq_hess(q, **kwargs):
    ia = IdealAnalytic(**kwargs)
    return np.array(
        [getattr(ia, q + "_%s_%s"%(x, y)) for x, y in ('hh', 'ha', 'hb', 'ha', 'aa', 'ab', 'hb', 'ab', 'bb')]).reshape(
        (3, 3))


def get_frac_q_jh(q, dx=1e-4, hess=False, **kwargs):
    num = ideal_numerical_jh(q, dx, hess, **kwargs)
    if hess:
        anl = np.squeeze(getq_hess(q, **kwargs))
    else:
        anl = np.squeeze(getq_jac(q, **kwargs))
    mask = np.logical_and(np.isclose(anl, 0, atol=dx), np.isclose(num, 0, atol=dx))
    anl = np.where(mask, 1, anl)

    num = np.where(mask, 1, num)
    return anl/num


# ------------ TEST BASIC FUNCTIONS -----------------------------------------------
def runq(q, hess, logHs, alpha, beta, lnA, logHsd, alphad, betad, lnAd):
    res = get_frac_q_jh(q, hess=hess, log_mmin=mmin, logHs=logHs,
                        alpha=alpha,lnA=lnA, beta=beta,logHsd=logHsd,
                        alphad=alphad, betad=betad,lnAd=lnAd)
    print res
    assert np.all(np.isclose(res,1.0, rtol=1e-2))


def test_basics():
    for q in ["_lng", "_lnq", "_u"]:
        for hess in [False, True]:
            for h, a, b,lnA, hd, ad, bd,lnAd in trials:
                yield runq, q, hess, h, a, b,lnA, hd, ad, bd, lnAd


# ------------ TEST G - Q -----------------------------------------------
def get_frac_gq_jh(dx=1e-4, hess=False, **kwargs):
    num = ideal_numerical_jh("_lng", dx, hess, **kwargs) - ideal_numerical_jh("_lnq", dx, hess, **kwargs)
    if hess:
        anl = np.squeeze(getq_hess("_lng", **kwargs)) - np.squeeze(getq_hess("_lnq", **kwargs))
    else:
        anl = np.squeeze(getq_jac("_lng", **kwargs)) - np.squeeze(getq_jac("_lnq", **kwargs))

    mask = np.logical_and(np.isclose(anl, 0, atol=dx), np.isclose(num, 0, atol=dx))
    anl = np.where(mask, 1, anl)
    num = np.where(mask, 1, num)
    return anl/num


def rungq(hess, logHs, alpha, beta,lnA, logHsd, alphad, betad, lnAd):
    res = get_frac_gq_jh(hess=hess, log_mmin=mmin, logHs=logHs,
                         alpha=alpha, beta=beta,lnA=lnA,logHsd=logHsd,
                         alphad=alphad, betad=betad,lnAd=lnAd)
    print res
    assert np.all(np.isclose(res,1.0, rtol=1e-2))


def test_gq():
    for hess in [False, True]:
        for h, a, b, lnA,hd, ad, bd,lnAd, in trials:
            yield rungq, hess, h, a, b,lnA, hd, ad, bd,lnAd


# ------------ TEST G - Q - U -----------------------------------------------
def get_frac_gqu_jh(dx=1e-4, hess=False, **kwargs):
    num = ideal_numerical_jh("_lng", dx, hess, **kwargs) - ideal_numerical_jh("_lnq", dx, hess,
                                                                              **kwargs) - ideal_numerical_jh("_u", dx,
                                                                                                             hess,
                                                                                                             **kwargs)
    if hess:
        anl = np.squeeze(getq_hess("_lng", **kwargs)) - np.squeeze(getq_hess("_lnq", **kwargs)) - np.squeeze(
            getq_hess("_u", **kwargs))
    else:
        anl = np.squeeze(getq_jac("_lng", **kwargs)) - np.squeeze(getq_jac("_lnq", **kwargs)) - np.squeeze(
            getq_jac("_u", **kwargs))

    mask = np.logical_and(np.isclose(anl, 0, atol=dx), np.isclose(num, 0, atol=dx))
    anl = np.where(mask, 1, anl)
    num = np.where(mask, 1, num)
    return anl/num


def rungqu(hess, logHs, alpha, beta,lnA, logHsd, alphad, betad, lnAd):
    res = get_frac_gqu_jh(hess=hess, log_mmin=mmin, logHs=logHs,
                          alpha=alpha, beta=beta,lnA=lnA,logHsd=logHsd,
                          alphad=alphad, betad=betad, lnAd=lnAd)
    print res
    assert np.all(np.isclose(res,1.0, rtol=1e-2))


def test_gqu():
    for hess in [False, True]:
        for h, a, b, lnA, hd, ad, bd,lnAd in trials:
            yield rungqu, hess, h, a, b,lnA, hd, ad, bd,lnAd


# ------------ TEST F -----------------------------------------------
def getF_jac(**kwargs):
    ia = IdealAnalytic(**kwargs)
    return np.array([ia._F_x(x) for x in 'hab'])


def getF_hess(**kwargs):
    ia = IdealAnalytic(**kwargs)
    return np.array([ia._F_x_y(x, y) for x, y in ('hh', 'ha', 'hb', 'ha', 'aa', 'ab', 'hb', 'ab', 'bb')]).reshape(
        (3, 3))


def get_frac_F_jh(dx=1e-4, hess=False, **kwargs):
    num = ideal_numerical_jh("_F", dx, hess, **kwargs)
    if hess:
        anl = np.squeeze(getF_hess(**kwargs))
    else:
        anl = np.squeeze(getF_jac(**kwargs))
    mask = np.logical_and(np.isclose(anl, 0, atol=dx), np.isclose(num, 0, atol=dx))
    anl = np.where(mask, 1, anl)

    num = np.where(mask, 1, num)
    return anl,num


def runF(hess, logHs, alpha, beta, lnA, logHsd, alphad, betad, lnAd):
    anl,num = get_frac_F_jh(hess=hess, log_mmin=mmin, logHs=logHs,
                        alpha=alpha, beta=beta,lnA=lnA,logHsd=logHsd,
                        alphad=alphad, betad=betad,lnAd=lnAd)
    print anl,num
    assert np.all(np.isclose(anl,num, rtol=3e-2,atol=1e-5))


def test_nablaF():
    for hess in [False, True]:
        for h, a, b, lnA, hd, ad, bd, lnAd in trials:
            # The following fails at the moment, so continue
            #if not hess and h==hd and a==ad and b==bd:
            #    continue

            yield runF, hess, h, a, b, lnA, hd, ad, bd, lnAd

