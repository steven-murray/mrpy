import inspect
import os

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys

sys.path.insert(0, LOCATION)

from mrpy.extra.likelihoods import SampleLike, CurveLike, SampleLikeWeights
from mrpy._utils import numerical_hess, numerical_jac
import numpy as np
from mrpy.base.core import dndm


# HELPER FUNCTIONS
def numerical_jh(cl, q, dx=1e-4, hess=False, **kwargs):
    def func(**kwargs):
        return getattr(cl(**kwargs), q)

    if hess:
        return numerical_hess(func, ["logHs", "alpha", 'beta', 'lnA'], dx, **kwargs)
    else:
        return numerical_jac(func, ["logHs", "alpha", 'beta', 'lnA'], dx, **kwargs)


def numerical_jh4(cl, q, dx=1e-4, hess=False, **kwargs):
    # This is necessary for CurveLike when all 4 params are free
    def func(**kwargs):
        return getattr(cl(**kwargs), q)

    if hess:
        return numerical_hess(func, ["logHs", "alpha", 'beta', "lnA"], dx, **kwargs)
    else:
        return numerical_jac(func, ["logHs", "alpha", 'beta', 'lnA'], dx, **kwargs)


# Set some default parameters to use throughout
alpha = -1.8
beta = 0.7
logHs = 14.0
mmin = 11.0
A = 1.0

# A series of trial parameters
trials = [(logHs, alpha, beta),
          (logHs * 1.1, alpha, beta),
          (logHs, alpha * 1.05, beta),
          (logHs, alpha, beta * 1.1)]


def get_jhq(cl, q, hess=False):
    """
    Gets the analytic jacobian or hessian from a given class, for a given quantity q.
    """
    sub = "hess" if hess else "jac"
    if q != "lnL":
        s = "%s_%s" % (q, sub)
        if "__" in s:
            s = s.replace("__", "_") + "_"
            print s
        return getattr(cl, s)
    else:
        return getattr(cl, "hessian" if hess else "jacobian")


class TestPO(object):
    d = 1e-5

    def run_q(self, q, hess, logHs, alpha, beta):
        # Data. For speed, just generate a couple of masses.
        logm = np.array([14.0, 14.0, 15.0, 13.0, 12.5])

        c = SampleLike(logm=logm, logHs=logHs, alpha=alpha, beta=beta, lnA=0)

        num = numerical_jh4(SampleLike, q, self.d, hess=hess,
                            logm=logm, logHs=logHs, alpha=alpha, beta=beta, lnA=0)
        anl = get_jhq(c, q, hess)
        print anl / num
        if hess:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=1e-5))
        else:
            assert np.all(np.isclose(anl, num, rtol=1e-3, atol=1e-5))

    def test_jh(self):
        for q in ["lnL", "_lng_", "_q_"]:
            for hess in [False, True]:
                # for s in [0,1]:
                for h, a, b in trials:
                    yield self.run_q, q, hess, h, a, b

    def test_eq_mmin(self):
        """
        Simply tests if the extra underscore quantities are defined the right way.
        """
        c = SampleLike(logm=np.array([14.0]), logHs=logHs, alpha=alpha,
                       beta=beta, lnA=0)
        for q in ["lng", 'q']:
            for p in ['h', 'a', 'b',
                      'h_h', 'h_a', 'h_b',
                      'a_a', 'a_b', 'b_b']:
                yield asseq, getattr(c, "_%s_%s" % (q, p)), getattr(c, "_%s_%s_" % (q, p))


def asseq(a, b):
    assert a == b


class TestCurve(object):
    d = 1e-5

    # def run_jh(self, q, hess, logHs, alpha, beta, rhomean):  # ,scale):
    #     # Data
    #     m = np.logspace(10, 15, 5)
    #     dn = dndm(m, logHs=logHs * 1.1, alpha=alpha * 1.05, beta=beta * 1.05, norm=A)
    #
    #     c = CurveLike(np.linspace(10, 15, 5), hs=logHs, alpha=alpha, beta=beta, lnA=np.log(A),
    #                   data_dndm=dn,  sig_rhomean=rhomean)
    #
    #     num = numerical_jh(CurveLike, q, self.d, hess=hess,
    #                        logm=np.linspace(10, 15, 5), hs=logHs, alpha=alpha, beta=beta,
    #                        data_dndm=dn, sig_rhomean=rhomean, lnA=np.log(A))
    #     anl = get_jhq(c, q, hess)
    #     print anl , num
    #     if hess:
    #         assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))
    #     else:
    #         assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))

    def run_jh4(self, hess, logHs, alpha, beta, A, rhomean):  # ,scale):
        # Data
        m = np.logspace(10, 15, 5)
        dn = dndm(m, logHs=logHs * 1.1, alpha=alpha * 1.05, beta=beta * 1.05, norm=A)

        c = CurveLike(np.linspace(10, 15, 5), logHs=logHs, alpha=alpha, beta=beta, lnA=np.log(A),
                      data_dndm=dn, sig_rhomean=rhomean, )

        num = numerical_jh4(CurveLike, "lnL", self.d, hess=hess,
                            logm=np.linspace(10, 15, 5), logHs=logHs, alpha=alpha, beta=beta,
                            data_dndm=dn, sig_rhomean=rhomean, lnA=np.log(A))
        anl = get_jhq(c, "lnL", hess)
        print anl , num
        if hess:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))
        else:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))

    # def test_jh(self):
    #     for q in ["lnL", "_lnk"]:
    #         for hess in [False]:  # hessians don't work here yet.
    #             for rhomean in [0, np.inf]:
    #                 # for s in (0,1):
    #                 for h, a, b in trials:
    #                     yield self.run_jh, q, hess, h, a, b, rhomean  # ,s

    def test_jh4(self):
        for hess in [False, True]:
            for rhomean in [np.inf, 1]:
                # for s in (0,1):
                for h, a, b in trials:
                    yield self.run_jh4, hess, h, a, b, A, rhomean  # ,s


def test_weights():
    m = np.array([14.0] * 3 + [15.0] * 2)
    m_unique = np.array([14, 15])
    m_counts = np.array([3, 2])

    standard = SampleLike(logm=m, logHs=14.0, alpha=-1.8, beta=0.75, lnA=0)
    weighted = SampleLikeWeights(logm=m_unique, weights=m_counts, logHs=14.0, alpha=-1.8, beta=0.75, lnA=0)

    print standard.lnL, weighted.lnL
    assert np.isclose(standard.lnL, weighted.lnL, rtol=1e-8)
