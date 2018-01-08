import numpy as np
from mrpy.extra.likelihoods import SampleLike, CurveLike

import mrpy.extra.reparameterise as repar

## base data
logm = np.array([12])
dndm = np.array([1.0])
mmin = 10.0
hs = 14.0
a = -1.8
b = 0.7

mrp_po = SampleLike(logm,hs,a,b,log_mmin=mmin, lnA=1.0)
mrp_curve = CurveLike(logm,dndm, hs,a,b,lnA=1.0)

class Base(object):
    cl = None
    lnl = np.ones(1)
    jac = None
    hess = None

    def __init__(self):
        self.inst = self.cl(logHs=hs,alpha=a,beta=b,logm=logm,log_mmin=mmin,lnA=1.0)

    def test_lnL(self):
        assert np.all(np.isclose(mrp_po.lnL/self.inst.lnL,self.lnl))
    #
    # def test_jac(self):
    #     assert np.all(np.isclose(self.inst.this_jacobian[:3]/mrp_po.jacobian[:3],self.jac))
    #
    # def test_hess(self):
    #     assert np.all(np.isclose(self.inst.this_hessian[:3,:3]/mrp_po.hessian[:3,:3],self.hess))


class TestAp1(Base):
    cl = repar.AP1Sample
    lnl = np.ones(1)
    jac = np.ones(3)
    hess = np.ones((3,3))

class TestGG2(Base):
    cl = repar.GG2Sample
    jac = np.array([-1.42857143, 1.,-6.31563321])
    hess = np.array([[  2.04081633,  -1.42857143,  13.6054405 ],
                     [ -1.42857143,   1.,         -10.28467647],
                     [ 13.6054405,   -10.28467647, 44.59361725]])

class TestGG3(Base):
    cl = repar.GG3Sample
    jac = np.array([  1., 0.7, -25.8328367])
    hess = np.array([[  1.,           0.7,          0.79346436],
                     [  0.7,          0.49,       -88.87404701],
                     [  0.79346436, -88.87404701,   4.01985689]])

class TestHT(Base):
    cl = repar.HTSample
    jac = np.array([ 1. ,0.95167147,  0.91805478])
    hess = np.array([[  1.,          -8.4412689,    0.87678067],
                     [ -8.4412689,    0.88979042, -12.23387305],
                     [  0.87678067, -12.23387305,   0.94268744]])

