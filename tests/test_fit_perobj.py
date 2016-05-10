import numpy as np
from mrpy.fit_perobj import PerObjFit
from mrpy.core import dndm, A_rhoc, _getnorm
from mrpy.stats import TGGD
from nose.tools import nottest

np.random.seed(42)

def test_s0():
    np.random.seed(42)
    t = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12)
    r = t.rvs(3e4)

    FitObj = PerObjFit(r)#,weight_scale=0)
    res,po = FitObj.run_downhill(hs0=14, alpha0=-1.8, beta0=1.0)

    print res.x
    assert res.success
    assert np.all(np.isclose(res.x[:3],[14.0,-1.8,1.0],rtol=5e-2))

def test_weighted():
    np.random.seed(42)
    t = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12)
    r = t.rvs(5e7)

    counts,bins = np.histogram(np.log10(r),bins=10000)
    FitObj = PerObjFit(10**bins[:-1][counts>0],nm=counts[counts>0])#,weight_scale=0)
    res,po = FitObj.run_downhill(hs0=14, alpha0=-1.8, beta0=1.0)

    print res.x
    assert res.success
    assert np.all(np.isclose(res.x[:3],[14.0,-1.8,1.0],rtol=5e-2))

def test_multisim():
    np.random.seed(42)
    "Test passing multiple sim boxes"
    t = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12)
    r = t.rvs(3e4)

    FitObj = PerObjFit(np.concatenate((r,r)))#,weight_scale=0)
    res1,po1 = FitObj.run_downhill(hs0=14, alpha0=-1.8, beta0=1.0)

    r = [r,r]
    FitObj = PerObjFit(r)#,weight_scale=0)
    res,po = FitObj.run_downhill(hs0=14, alpha0=-1.8, beta0=1.0)

    print res,res1
    assert res.success
    #assert np.isclose(res1.x[-1],res.x[-1] - np.log(2),rtol=5e-2)
    assert np.all(np.isclose(res.x,res1.x,rtol=5e-2))



# def test_weighted_nu2():
#     t = TGGD(scale=1e14,a=-1.9,b=0.8,xmin=1e10)
#     r = t.rvs(5.27e7)
#
#
#     counts,bins = np.histogram(r,bins=np.arange(8.8e9,1e15,2.2e8))
#     FitObj = PerObjFit(bins[:-1][counts>0],nm=counts[counts>0])#,weight_scale=0)
#     res,po = FitObj.run_downhill(hs0=14, alpha0=-1.9, beta0=0.8)
#
#     print res.x
#     assert res.success
#     assert np.all(np.isclose(res.x[:3],[14.0,-1.9,0.8],rtol=5e-2))



# def test_s1():
#     t = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12)
#     r = t.rvs(5e5)
#     FitObj = PerObjFit(r,weight_scale=1.0,alpha_bounds=(-1.95,-1.7), hs_bounds=(10,16),
#                                  beta_bounds=(0.8,1.2))
#     res,po = FitObj.run_downhill(hs0=14, alpha0=-1.8, beta0=1.0)
#
#     print res.x
#     assert res.success
#     assert np.all(np.isclose(res.x,[14.0,-1.8,1.0],rtol=5e-2))