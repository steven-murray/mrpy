import numpy as np
from mrpy.fit_perobj import fit_perobj_opt
from mrpy.core import dndm, A_rhoc, _getnorm
from mrpy.stats import TGGD
from nose.tools import nottest

np.random.seed(42)

def test_s0():
    t = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12)
    r = t.rvs(1e5)

    res,po = fit_perobj_opt(r, hs0=14, alpha0=-1.8, beta0=1.0, s=0)

    print res.x
    assert res.success
    assert np.all(np.isclose(res.x,[14.0,-1.8,1.0],rtol=5e-2))


def test_s1():
    t = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12)
    r = t.rvs(4e5)
    print np.log10(r).min(), np.log10(r).max()
    res,po = fit_perobj_opt(r, hs0=14, alpha0=-1.8, beta0=1.0, s=1,
                            alpha_bounds=(-1.95,-1.7), hs_bounds=(10,16),
                            beta_bounds=(0.8,1.2))

    print res.x
    assert res.success
    assert np.all(np.isclose(res.x,[14.0,-1.8,1.0],rtol=5e-2))

@nottest
def test_bounds():
    t = TGGD(scale=1e14, a=-1.8, b=1.0, xmin=1e12)
    r = t.rvs(1e5)

    res,po = fit_perobj_opt(r, hs0=14, alpha0=-1.8, beta0=1.0, s=0,
                            bounds=None)

    print res.x
    assert res.success