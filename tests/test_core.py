import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from scipy.integrate import simps
rd = np.random.random
from mrpy import core, stats
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def test_pdf_inf():
    """
    Test for 10 random values of the MRP parameters, whether the integral of the
    pdf is 1.
    """
    m = np.logspace(10,17,500)
    for i in range(10):
        ans =  simps(core.dndm(m,3*rd()+10,0.2*rd()-2.0,0.5*rd()+0.5,norm="pdf"),m)
        err = np.abs(ans-1)
        print err
        assert np.abs(ans-1) < 1e-4

# def test_pdf_mmax():
#     """
#     Test for 10 random values of the MRP parameters, whether the integral of the
#     pdf is 1.
#     """
#     m = np.logspace(10,17,500)
#     for i in range(10):
#         ans =  simps(core.mrp(m,3*rd()+10,0.2*rd()-2.0,0.5*rd()+0.5,norm="pdf",mmax=np.log10(m[-1])),m)
#         err = np.abs(ans-1)
#         print err
#         assert np.abs(ans-1) < 1e-4

def test_Arhoc():
    # Try A_rhoc
    m = np.logspace(0,17,2000)
    for i in range(100):
        ans =  simps(core.dndm(m,3*rd()+10,-1.5,0.5*rd()+0.5,norm="rhoc",Om0=0.3)*m,m)/(0.3*2.7755e11)
        err = np.abs(ans-1)
        print err
        assert np.abs(ans-1) < 1e-4


def test_ngtm_pdf():
    """
    Make sure the cdf is 1 at mmin
    """
    m = np.logspace(10,12,20)
    assert core.ngtm(m,14.0,-1.9,0.8)[0] == 1


def test_log_mass_mode():
    lmm = core.log_mass_mode(14.0,-1.8,0.7)
    m = np.linspace(np.log10(lmm)-1,np.log10(lmm)+1,200)
    mrp = stats.TGGDlog(14.0,-1.8+1,0.7,m[0]).pdf(m,log=True)
    s = spline(m,mrp,k=4)
    assert np.isclose(lmm,10**s.derivative().roots())

class TestMRP(object):
    def __init__(self):
        logm = np.linspace(10,15,500)
        self.mrp = core.MRP(logm,14.0,-1.9,0.75)

    def test_m(self):
        assert np.all(self.mrp.m == np.logspace(10,15,500))

    def test_mmin(self):
        assert np.all(self.mrp.mmin == 10**10)

    def test_Hs(self):
        assert np.all(self.mrp.Hs == 10**14.0)

    def test_pdf(self):
        assert np.isclose(simps(self.mrp.dndm(),self.mrp.m),1.0)

    def test_nbar(self):
        print self.mrp.A, self.mrp.lnA
        print self.mrp.nbar
        assert np.isclose(self.mrp.nbar,1.0)

    def test_nbar_general_m(self):
        logm = np.linspace(11,15,100)
        mrp = core.MRP(logm,14.0,-1.9,0.75,log_mmin=10.0)
        assert np.isclose(mrp.nbar,1.0)

    def test_arhoc(self):
        logm = np.linspace(0,16,1000)
        mrp = core.MRP(logm,14.0,-1.85,0.75,norm="rhoc")
        print mrp.rhobar/(0.3*2.7755e11)
        assert np.isclose(mrp.rhobar,0.3*2.7755e11,rtol=0.05)


    def test_arhoc_general_m(self):
        logm = np.linspace(10,16,1000)
        mrp = core.MRP(logm,14.0,-1.85,0.75,norm="rhoc",log_mmin=0.0)
        print mrp.rhobar/(0.3*2.7755e11)
        assert np.isclose(mrp.rhobar,0.3*2.7755e11,rtol=0.05)

    def test_lmm(self):
        lmm = self.mrp.log_mass_mode
        m = np.linspace(np.log10(lmm)-1,np.log10(lmm)+1,200)
        mrp = stats.TGGDlog(14.0,-1.9+1,0.75,m[0]).pdf(m,log=True)
        s = spline(m,mrp,k=4)
        assert np.isclose(lmm,10**s.derivative().roots())

