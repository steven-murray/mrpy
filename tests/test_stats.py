"""
Basic tests of the statistics.
"""
import inspect
import os

import numpy as np

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)


from mrpy.base.stats import TGGD, TGGDlog, TGGDln
import numpy as np
from scipy.integrate import quad
from scipy.stats import skew, kurtosis



class TestTGGD(object):
    def __init__(self):
        np.random.seed(1234)
        self.tggd = TGGD(a=-1.5,b=0.7,xmin=1e10,scale=1e14)
        self.rsmall = self.tggd.rvs(100)
        self.rlarge = self.tggd.rvs(1e6)

    def test_cdf_quantile(self):
        a = self.tggd.quantile(self.tggd.cdf(self.rsmall))/self.rsmall
        assert np.all(np.isclose(a,1))

    def test_pdf_integrate(self):
        a = quad(self.tggd.pdf,1e10,1e11)[0]/self.tggd.cdf(1e11)
        assert np.isclose(a,1)

    def test_large_cdf(self):
        a = self.tggd.cdf(1e18)
        assert np.isclose(a,1)

    def test_compare_log(self):
        tggd_log = TGGDlog(a=-1.5,b=0.7,xmin=10,scale=14)
        a = self.tggd.cdf(10**tggd_log.quantile(np.arange(0,1,0.1)))
        assert np.all(np.isclose(a,np.arange(0,1,0.1)))

    def test_compare_ln(self):
        tggd_ln = TGGDln(a=-1.5,b=0.7,xmin=np.log(1e10),scale=np.log(1e14))
        a = self.tggd.cdf(np.exp(tggd_ln.quantile(np.arange(0,1,0.1))))
        assert np.all(np.isclose(a,np.arange(0,1,0.1)))

    def test_mean(self):
        print "Sampled, Analytic: ", np.mean(self.rlarge), self.tggd.mean
        assert np.isclose(np.mean(self.rlarge)/self.tggd.mean,1,rtol=1e-2)

    def test_variance(self):
        print "Sampled, Analytic: ", np.var(self.rlarge), self.tggd.variance
        assert np.isclose(np.var(self.rlarge)/self.tggd.variance,1,rtol=1e-1)

    def test_skewness(self):
        print "Sampled, Analytic: ", skew(self.rlarge), self.tggd.skewness
        assert np.isclose(skew(self.rlarge)/self.tggd.skewness,1,rtol=1e-1)

    # Can't do the following because the samples are too poor.
    # def test_kurtosis(self):
    #     print "Sampled, Analytic: ", kurtosis(self.rlarge), self.tggd.kurtosis
    #     assert np.isclose(kurtosis(self.rlarge)/self.tggd.kurtosis,1,rtol=1e-1)

class TestTGGDlog(object):
    def __init__(self):
        np.random.seed(1234)
        self.tggd = TGGDlog(a=-1.5,b=0.7,xmin=10,scale=14)
        self.rsmall = self.tggd.rvs(100)

    def test_cdf_quantile(self):
        a = self.tggd.quantile(self.tggd.cdf(self.rsmall))/self.rsmall
        assert np.all(np.isclose(a,1))

    def test_pdf_integrate(self):
        a = quad(self.tggd.pdf,10,11)[0]/self.tggd.cdf(11)
        assert np.isclose(a,1)

    def test_large_cdf(self):
        a = self.tggd.cdf(18)
        assert np.isclose(a,1)

    def test_compare_real(self):
        tggd_real = TGGD(a=-1.5,b=0.7,xmin=1e10,scale=1e14)
        a = self.tggd.cdf(np.log10(tggd_real.quantile(np.arange(0,1,0.1))))
        assert np.all(np.isclose(a,np.arange(0,1,0.1)))

    def test_compare_ln(self):
        tggd_ln = TGGDln(a=-1.5,b=0.7,xmin=np.log(1e10),scale=np.log(1e14))
        a = self.tggd.cdf(tggd_ln.quantile(np.arange(0,1,0.1))/np.log(10))
        assert np.all(np.isclose(a,np.arange(0,1,0.1)))


class TestTGGDln(object):
    def __init__(self):
        np.random.seed(1234)
        self.tggd = TGGDln(a=-1.5,b=0.7,xmin=np.log(1e10),scale=np.log(1e14))
        self.rsmall = self.tggd.rvs(100)

    def test_cdf_quantile(self):
        a = self.tggd.quantile(self.tggd.cdf(self.rsmall))/self.rsmall
        assert np.all(np.isclose(a,1))

    def test_pdf_integrate(self):
        a = quad(self.tggd.pdf,np.log(1e10),np.log(1e11))[0]/self.tggd.cdf(np.log(1e11))
        assert np.isclose(a,1)

    def test_large_cdf(self):
        a = self.tggd.cdf(np.log(1e18))
        assert np.isclose(a,1)

    def test_compare_real(self):
        tggd_real = TGGD(a=-1.5,b=0.7,xmin=1e10,scale=1e14)
        a = self.tggd.cdf(np.log(tggd_real.quantile(np.arange(0,1,0.1))))
        assert np.all(np.isclose(a,np.arange(0,1,0.1)))

    def test_compare_log(self):
        tggd_log = TGGDlog(a=-1.5,b=0.7,xmin=10,scale=14)
        a = self.tggd.cdf(tggd_log.quantile(np.arange(0,1,0.1))*np.log(10))
        assert np.all(np.isclose(a,np.arange(0,1,0.1)))
