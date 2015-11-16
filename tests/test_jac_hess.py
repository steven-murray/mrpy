import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)

from mrpy.likelihoods import MRP_PO_Likelihood

## Define numerical jac/hess
def numerical_jac(logm,hs,alpha,beta,scale=0,delta=0.0001):
    base = MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha,beta=beta)
    return np.array([MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs+delta,alpha=alpha,beta=beta).lnL-base.lnL,
                     MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha+delta,beta=beta).lnL-base.lnL,
                     MRP_PO_Likelihood(scale=scale,logm = logm,logHs=hs,alpha=alpha,beta=beta+delta).lnL-base.lnL])/delta

def numerical_hess(logm,hs,alpha,beta,scale=0,delta=0.0001):
    jac0 = numerical_jac(logm,hs,alpha,beta,scale,delta)
    return np.array([numerical_jac(logm,hs+delta,alpha,beta,scale,delta)-jac0,
                     numerical_jac(logm,hs,alpha+delta,beta,scale,delta)-jac0,
                     numerical_jac(logm,hs,alpha,beta+delta,scale,delta)-jac0])/delta

def run_jac(hs,alpha,beta,s):
    m = np.arange(10,17,500)
    mrp = MRP_PO_Likelihood(scale=s,logm = m,logHs=hs,alpha=alpha,beta=beta)
    print mrp.jacobian
    print numerical_jac(m,hs,alpha,beta,s)
    assert np.all(np.isclose(mrp.jacobian,numerical_jac(m,hs,alpha,beta,s),rtol=1e-3,atol=0))

def run_hess(hs,alpha,beta,s):
    m = np.arange(10,17,500)
    mrp = MRP_PO_Likelihood(scale=s,logm = m,logHs=hs,alpha=alpha,beta=beta)
    print mrp.hessian
    print numerical_hess(m,hs,alpha,beta,s)
    assert np.all(np.isclose(mrp.hessian,numerical_hess(m,hs,alpha,beta,s),rtol=1e-2,atol=0))


def test_jac_14_19_8_0():
    run_jac(14.0,-1.9,0.8,0)

def test_jac_10_18_2_0():
    run_jac(10.0,-1.8,0.2,0)

def test_jac_14_19_8_1():
    run_jac(14.0,-1.9,0.8,1)

def test_jac_14_19_8_12():
    run_jac(14.0,-1.9,0.8,1.2)

def test_hess_14_19_8_0():
    run_hess(14.0,-1.9,0.8,0)

def test_hess_10_18_2_0():
    run_hess(10.0,-1.8,0.2,0)

def test_hess_14_19_8_1():
    run_hess(14.0,-1.9,0.8,1)

def test_hess_14_19_8_12():
    run_hess(14.0,-1.9,0.8,1.2)
