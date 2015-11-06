import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from scipy.integrate import simps
rd = np.random.random
from mrpy import core

def test_pdf_inf():
    """
    Test for 100 random values of the MRP parameters, whether the integral of the
    pdf is 1.
    """
    m = np.logspace(10,17,500)
    for i in range(100):
        ans =  simps(core.mrp(m,3*rd()+10,0.2*rd()-2.0,0.5*rd()+0.5,norm="pdf"),m)
        err = np.abs(ans-1)
        print err
        assert np.abs(ans-1) < 1e-4

def test_pdf_mmax():
    """
    Test for 100 random values of the MRP parameters, whether the integral of the
    pdf is 1.
    """
    m = np.logspace(10,17,500)
    for i in range(100):
        ans =  simps(core.mrp(m,3*rd()+10,0.2*rd()-2.0,0.5*rd()+0.5,norm="pdf",mmax=np.log10(m[-1])),m)
        err = np.abs(ans-1)
        print err
        assert np.abs(ans-1) < 1e-4

def test_Arhoc():
    # Try A_rhoc
    m = np.logspace(0,17,2000)
    for i in range(100):
        ans =  simps(core.mrp(m,3*rd()+10,-1.5,0.5*rd()+0.5,norm="rhoc",Om0=0.3)*m,m)/(0.3*2.7755e11)
        err = np.abs(ans-1)
        print err
        assert np.abs(ans-1) < 1e-4
