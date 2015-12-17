import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)

from mrpy import mrp
from mrpy.likelihoods import MRP_PO_Likelihood,MRP_Curve_Likelihood
import numpy as np

## Define numerical jac/hess
def numerical_jac_po(logm,hs,alpha,beta,scale=0,dx=0.0001):
    base = MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha,beta=beta)
    return np.array([MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs+dx,alpha=alpha,beta=beta).lnL-base.lnL,
                     MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha+dx,beta=beta).lnL-base.lnL,
                     MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha,beta=beta+dx).lnL-base.lnL])/dx

def numerical_hess_po(logm,hs,alpha,beta,scale=0,dx=0.0001):
    jac0 = numerical_jac_po(logm,hs,alpha,beta,scale,delta)
    return np.array([numerical_jac_po(logm,hs+dx,alpha,beta,scale,dx)-jac0,
                     numerical_jac_po(logm,hs,alpha+dx,beta,scale,dx)-jac0,
                     numerical_jac_po(logm,hs,alpha,beta+dx,scale,dx)-jac0])/dx

class TestPO(object):
    def __init__(self):
        # Parameters used to construct data
        self.hs_def = 14.0
        self.alpha_def = -1.9
        self.beta_def = 0.8
        self.lnA_def = 0

        # bad way to make data!
        self.logm = np.linspace(10,16,200)

        #Parameters used to get jac/hess
        self.hs_sol = 14.2
        self.alpha_sol = -1.85
        self.beta_sol = 0.75
        self.lnA_sol = 1

        self.d = 1e-5
        def run_jac(self,s):
            c = MRP_PO_Likelihood(scale=s,logm = self.logm,logHs=self.hs_sol,
                                  alpha=self.alpha_sol,beta=self.beta_sol)

            num = numerical_jac_po(self.logm,self.hs_sol,self.alpha_sol,self.beta_sol,s,self.d)
            print c.jacobian/num - 1
            assert np.all(np.isclose(mrp.jacobian,num,rtol=1e-3,atol=0))

        def run_hess(self,s):
            c = MRP_PO_Likelihood(scale=s,logm = self.logm,logHs=self.hs_sol,
                                  alpha=self.alpha_sol,beta=self.beta_sol)

            num = numerical_hess_po(self.logm,self.hs_sol,self.alpha_sol,self.beta_sol,s,self.d)
            print c.hessian/num -1
            assert np.all(np.isclose(mrp.hessian,num,rtol=1e-3,atol=0))

        def test_jac(self):
            for s in range(3):
                yield self.run_jac, s

        def test_hess(self):
            for s in range(3):
                yield self.run_hess, s


def numerical_jac_curve(logm, dndm,logHs,alpha,beta,lnA,sig_rhomean=np.inf,sig_integ=np.inf,sig_data=1,scale=0,dx=0.0001):
    y0 = MRP_Curve_Likelihood(logm = logm, dndm=dndm,sig_rhomean=sig_rhomean,sig_integ=sig_integ,sig_data=sig_data,
                              alpha=alpha,beta=beta,logHs=logHs,lnA=lnA,scale=scale).lnL
    return np.array([MRP_Curve_Likelihood(logm = logm, dndm=dndm,sig_rhomean=sig_rhomean,sig_integ=sig_integ,sig_data=sig_data,
                              alpha=alpha,beta=beta,logHs=logHs+dx,lnA=lnA,scale=scale).lnL-y0,
                  MRP_Curve_Likelihood(logm = logm,dndm=dndm,sig_rhomean=sig_rhomean,sig_integ=sig_integ,sig_data=sig_data,
                              alpha=alpha+dx,beta=beta,logHs=logHs,lnA=lnA,scale=scale).lnL-y0,
                  MRP_Curve_Likelihood(logm = logm, dndm=dndm,sig_rhomean=sig_rhomean,sig_integ=sig_integ,sig_data=sig_data,
                              alpha=alpha,beta=beta+dx,logHs=logHs,lnA=lnA,scale=scale).lnL-y0,
                 MRP_Curve_Likelihood(logm = logm, dndm=dndm,sig_rhomean=sig_rhomean,sig_integ=sig_integ,sig_data=sig_data,
                              alpha=alpha,beta=beta,logHs=logHs,lnA=lnA+dx,scale=scale).lnL-y0])/dx

def numerical_hess_curve(logm, dndm,logHs,alpha,beta,lnA,sig_rhomean=np.inf,sig_integ=np.inf,sig_data=1,scale=0,dx=0.0001):
    y0 = numerical_jac_curve(logm, dndm,logHs,alpha,beta,lnA,sig_rhomean,sig_integ,sig_data,scale,dx)
    return np.array([numerical_jac_curve(logm, dndm,logHs+dx,alpha,beta,lnA,sig_rhomean,sig_integ,sig_data,scale,dx)-y0,
                  numerical_jac_curve(logm, dndm,logHs,alpha+dx,beta,lnA,sig_rhomean,sig_integ,sig_data,scale,dx)-y0,
                  numerical_jac_curve(logm, dndm,logHs,alpha,beta+dx,lnA,sig_rhomean,sig_integ,sig_data,scale,dx)-y0,
                  numerical_jac_curve(logm, dndm,logHs,alpha,beta,lnA+dx,sig_rhomean,sig_integ,sig_data,scale,dx)-y0])/dx


class TestCurve(object):
    def __init__(self):
        # Parameters used to construct data
        self.hs_def = 14.1
        self.alpha_def = -1.88
        self.beta_def = 0.78
        self.lnA_def = -42.5

        self.logm = np.linspace(10,16,200)
        self.dndm = mrp(10**self.logm,self.hs_def,self.alpha_def,self.beta_def,norm=np.exp(self.lnA_def))

        #Parameters used to get jac/hess
        self.hs_sol = 14.0
        self.alpha_sol = -1.9
        self.beta_sol = 0.8
        self.lnA_sol = -43

        self.d = 1e-5

    def run_jac(self,integ,rhomean,s):
        c = MRP_Curve_Likelihood(logm = self.logm,dndm=self.dndm,
                                 logHs=self.hs_sol,lnA=self.lnA_sol,
                                 alpha=self.alpha_sol,beta=self.beta_sol,
                                 sig_integ=integ,sig_rhomean=rhomean,scale=s)

        num = numerical_jac_curve(self.logm, self.dndm,self.hs_sol,self.alpha_sol,
                                  self.beta_sol,self.lnA_sol,rhomean,integ,
                                  scale=s,dx=self.d)[:len(c.jacobian)]

        #print "Analytic: ", c.jacobian
        #print "Numerical: ", num
        print c.jacobian/num -1
        assert np.all(np.isclose(c.jacobian,num,rtol=5e-3,atol=0))

    def run_hess(self,integ,rhomean,s):
        c = MRP_Curve_Likelihood(logm = self.logm,dndm=self.dndm,
                                 logHs=self.hs_sol,lnA=self.lnA_sol,
                                 alpha=self.alpha_sol,beta=self.beta_sol,
                                 sig_integ=integ,sig_rhomean=rhomean,scale=s)

        num = numerical_hess_curve(self.logm, self.dndm,self.hs_sol,self.alpha_sol,
                                   self.beta_sol,self.lnA_sol,rhomean,integ,
                                   scale=s,dx=self.d)[:c.hessian.shape[0],:c.hessian.shape[1]]

        print c.hessian/num -1
        assert np.all(np.isclose(c.hessian,num,rtol=5e-2,atol=0))

    def test_jac(self):
        for integ,rhomean,s in [(np.inf,np.inf,0),(np.inf,0,0),(0,np.inf,0),
                                (np.inf,np.inf,1),(np.inf,0,1),(0,np.inf,1),
                                (1,1,1),(1,1,0)]:
            yield self.run_jac, integ, rhomean,s

    def test_hess(self):
        for integ,rhomean,s in [(np.inf,np.inf,0),(np.inf,np.inf,1),
                                (1,1,1),(1,1,0)]:
            yield self.run_hess, integ, rhomean,s
