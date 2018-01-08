import numpy as np
from mrpy.base.core import dndm

from mrpy.fitting.fit_curve import get_fit_curve


## Basically the idea is to produce lots of different types of curves and fit
## them to see if they match up or errors occur.

def test_basic():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    res,_ = get_fit_curve(m,d,[14.0,-1.9,0.75,0.0])
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75,0.0])))

def test_shifted_start():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    res,_ = get_fit_curve(
        m,d,[14.1,-1.85,0.7,0.1],
        bounds=[(13.5,15.0), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)],
    )
    # For some reason, it seems I need the bounds on logHs above, or else I need to start it very close to the truth.
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75,0.0]), atol=1e-5))

def test_inf_0():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm="rhom")
    #a = A_rhoc(14.0,-1.9,0.75)

    res,_ = get_fit_curve(
        m,d,[14.5,-1.8, 0.7],
        sig_rhomean=0.0,
        bounds=[(10,16),(-1.99,-1.5), (-np.inf,np.inf)]
    )
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75])))

# def test_0_inf():
#     m = np.logspace(10,15,500)
#     d = dndm(m,14.0,-1.9,0.75,norm="pdf")
#
#     res,_ = get_fit_curve(
#         m,d,[14.5,-1.8,0.7, 0],
#         sigma_integ=0.0,
#         bounds=[(10,16),alpha_bounds=(-1.99,-1.5))
#     print res
#     assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75]),rtol=1e-3))


def test_no_jac():
    m = np.logspace(10,15,500)
    d = dndm(m,14.0,-1.9,0.75,norm=1.0)

    res,_ = get_fit_curve(m,d,[14.0,-1.9,0.75,0.0],jac=False)
    print res
    assert np.all(np.isclose(res.x,np.array([14.0,-1.9,0.75,0.0])))

