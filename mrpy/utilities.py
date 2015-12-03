"""
Functions to do various things with the MRP, like generate a sample of masses.
"""
import numpy as np
import core
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def generate_masses(N,hs,alpha,beta,mmin):
    """
    Return a list of halo masses sampled from current parameters
    """
    Hs = 10**hs


    mmax = Hs * (600.0) ** (1 / beta)

    m = 10 ** np.linspace(mmin, np.log10(mmax), 2000)
    cdf = core.ngtm(m,hs,alpha,beta,mmin=mmin,mmax=np.inf,norm="cdf",log=False)
    icdf = spline(cdf[::-1], np.log10(m[::-1]), k=2)

    x = np.random.random(N)
    y = 10 ** icdf(x)

    # Just to make sure everything's ok
    i = 1
    while len(y[y < 0]) > 0 or len(y[y > mmax]) > 0:
        i *= 2
        print "Increasing Resolution..."
        m = 10 ** np.linspace(np.log10(self.mmin), np.log10(mmax), 5000 * i)
        cdf = core.ngtm(m,hs,alpha,beta,mmin=mmin,mmax=np.inf,norm="cdf",log=False)
        icdf = spline(cdf[::-1], np.log10(m[::-1]), k=2)
        y = 10 ** icdf(x)

    return y
