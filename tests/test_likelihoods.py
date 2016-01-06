import inspect
import os

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys

sys.path.insert(0, LOCATION)

from mrpy import dndm
from mrpy.likelihoods import PerObjLike, CurveLike
from mrpy._utils import numerical_hess, numerical_jac
import numpy as np
from mrpy.stats import TGGD
from mrpy.core import dndm
from scipy.integrate import simps

## HELPER FUNCTIONS
def numerical_jh(cl, q, dx=1e-4, hess=False, **kwargs):
    def func(**kwargs):
        return getattr(cl(**kwargs), q)

    if hess:
        return numerical_hess(func, ["logHs", "alpha", 'beta'], dx, **kwargs)
    else:
        return numerical_jac(func, ["logHs", "alpha", 'beta'], dx, **kwargs)

def numerical_jh4(cl, q, dx=1e-4, hess=False, **kwargs):
    # This is necessary for CurveLike when all 4 params are free
    def func(**kwargs):
        return getattr(cl(**kwargs), q)

    if hess:
        return numerical_hess(func, ["logHs", "alpha", 'beta',"norm"], dx, **kwargs)
    else:
        return numerical_jac(func, ["logHs", "alpha", 'beta','norm'], dx, **kwargs)

## Set some default parameters to use throughout
alpha = -1.8
beta = 0.7
logHs = 14.0
mmin = 11.0
A = 1.0

# A series of trial parameters
trials = [(logHs, alpha, beta),
          (logHs*1.1, alpha, beta),
          (logHs, alpha*1.05, beta),
          (logHs, alpha, beta*1.1)]


#
# ## Define numerical jac/hess
# def numerical_jac_po(logm,hs,alpha,beta,scale=0,dx=0.0001):
#     return numerical_jac()
#     base = MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha,beta=beta)
#     return np.array([MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs+dx,alpha=alpha,beta=beta).lnL-base.lnL,
#                      MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha+dx,beta=beta).lnL-base.lnL,
#                      MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha,beta=beta+dx).lnL-base.lnL])/dx
#
# def numerical_hess_po(logm,hs,alpha,beta,scale=0,dx=0.0001):
#     jac0 = numerical_jac_po(logm,hs,alpha,beta,scale,delta)
#     return np.array([numerical_jac_po(logm,hs+dx,alpha,beta,scale,dx)-jac0,
#                      numerical_jac_po(logm,hs,alpha+dx,beta,scale,dx)-jac0,
#                      numerical_jac_po(logm,hs,alpha,beta+dx,scale,dx)-jac0])/dx

def get_jhq(cl,q,hess=False):
    """
    Gets the analytic jacobian or hessian from a given class, for a given quantity q.
    """
    sub = "hess" if hess else "jac"
    if q != "lnL":
        s = "%s_%s"%(q,sub)
        if "__" in s:
            s = s.replace("__","_")+"_"
        return getattr(cl,s)
    else:
        return getattr(cl,"hessian" if hess else "jacobian")

class TestPO(object):
    d = 1e-5
    def run_q(self, q,hess, logHs, alpha, beta, scale):
        # Data. For speed, just generate a couple of masses.
        logm = np.array([14.0, 15.0])

        c = PerObjLike(scale=scale, logm=logm, logHs=logHs, alpha=alpha, beta=beta)

        num = numerical_jh(PerObjLike, q, self.d, hess=hess,
                           scale=scale, logm=logm, logHs=logHs, alpha=alpha, beta=beta)
        anl = get_jhq(c,q,hess)
        print anl/num
        if hess:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))
        else:
            assert np.all(np.isclose(anl, num, rtol=1e-3, atol=0))

    def test_jh(self):
        for q in ["lnL","_lng_","_lnq_"]:
            for hess in [False, True]:
                for s in [0,1]:
                    for h, a, b in trials:
                        yield self.run_q, q, hess, h, a, b, s

    def test_eq_mmin(self):
        """
        Simply tests if the extra underscore quantities are defined the right way.
        """
        c = PerObjLike(scale=1.0, logm=np.array([14.0]), logHs=logHs, alpha=alpha,
                       beta=beta)
        for q in ["lng",'lnq']:
            for p in ['h','a','b',
                      'h_h','h_a','h_b',
                      'a_a','a_b','b_b']:
                yield asseq, getattr(c,"_%s_%s"%(q,p)), getattr(c,"_%s_%s_"%(q,p))

def asseq(a,b):
    assert a==b

class TestCurve(object):
    d=1e-5

    def run_jh(self, q,hess, logHs, alpha, beta, integ,rhomean,scale):
        # Data
        m = np.logspace(10,15,5)
        dn = dndm(m,logHs=logHs*1.1, alpha=alpha*1.05, beta=beta*1.05,norm=A)
        mwinteg = simps(m*dn,dx=np.log10(m[1]/m[0]))

        c = CurveLike(np.linspace(10,15,5),logHs=logHs, alpha=alpha, beta=beta,
                      dndm=dn, sig_integ=integ,sig_rhomean=rhomean,mw_integ=mwinteg)

        num = numerical_jh(CurveLike, q, self.d, hess=hess,
                           logm=np.linspace(10,15,5),logHs=logHs, alpha=alpha, beta=beta,
                           dndm=dn,sig_integ=integ,sig_rhomean=rhomean,mw_integ=mwinteg)
        anl = get_jhq(c,q,hess)
        print anl/num
        if hess:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))
        else:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))


    def run_jh4(self, hess, logHs, alpha, beta, A, integ,rhomean,scale):
        # Data
        m = np.logspace(10,15,5)
        dn = dndm(m,logHs=logHs*1.1, alpha=alpha*1.05, beta=beta*1.05,norm=A)
        mwinteg = simps(m*dn,dx=np.log10(m[1]/m[0]))

        c = CurveLike(np.linspace(10,15,5),logHs=logHs, alpha=alpha, beta=beta,norm=A,
                      dndm=dn,sig_integ=integ,sig_rhomean=rhomean,mw_integ=mwinteg)

        num = numerical_jh4(CurveLike, "lnL", self.d, hess=hess,
                           logm=np.linspace(10,15,5),logHs=logHs, alpha=alpha, beta=beta,
                           dndm=dn,sig_integ=integ,sig_rhomean=rhomean,norm=A,mw_integ=mwinteg)
        anl = get_jhq(c,"lnL",hess)
        print anl/num
        if hess:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))
        else:
            assert np.all(np.isclose(anl, num, rtol=1e-2, atol=0))


    def test_jh(self):
        for q in ["lnL","_lnk"]:
            for hess in [False]: #hessians don't work here yet.
                for integ, rhomean in [(np.inf, 0), ( 0,np.inf)]:
                    for s in (0,1):
                        for h, a, b in trials:
                            yield self.run_jh,  q, hess, h, a, b, integ,rhomean,s

    def test_jh4(self):
        for hess in [False, True]:
            for integ, rhomean in [(np.inf,np.inf), (1,1)]:
                for s in (0,1):
                    for h, a, b in trials:
                        yield self.run_jh4,  hess, h, a, b, A, integ,rhomean,s
