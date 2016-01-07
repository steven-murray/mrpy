import numpy as np
from mrpy.fit_curve import get_fit_curve
from mrpy.core import dndm, A_rhoc, _getnorm


## Basically the idea is to produce lots of different types of curves and fit
## them to see if they match up or errors occur.

def test_basic():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    res,_ = get_fit_curve(m,d,hs0=14.0,alpha0=-1.9,beta0=0.75,lnA0=0.0)
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75,0.0])))

def test_shifted_start():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    res,_ = get_fit_curve(m,d,hs0=14.5,alpha0=-1.8,beta0=0.7,lnA0=0.5)
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75,0.0])))

def test_inf_0():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm="rhoc")
    #a = A_rhoc(14.0,-1.9,0.75)

    res,_ = get_fit_curve(m,d,hs0=14.5,alpha0=-1.8,beta0=0.7,
                    sigma_rhomean=0.0,
                    bounds=True,hs_bounds=(10,16),alpha_bounds=(-1.99,-1.5))
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75])))

def test_0_inf():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm="pdf")
    #a = _getnorm("pdf", 14.0,-1.9, 0.75,1e10, log=True)

    res,_ = get_fit_curve(m,d,hs0=14.5,alpha0=-1.8,beta0=0.7,
                    sigma_integ=0.0,
                    bounds=True,hs_bounds=(10,16),alpha_bounds=(-1.99,-1.5))
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75]),rtol=1e-3))


def test_no_jac():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    res,_ = get_fit_curve(m,d,hs0=14.0,alpha0=-1.9,beta0=0.75,lnA0=0.0,jac=False)
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75,0.0])))

