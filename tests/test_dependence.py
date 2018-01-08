"""
Test `physical_dependence` module.
"""

# Hard to do this properly without import hmf code, which I don't really want to do.
# Perhaps for now, we'll just run things through to make sure they don't error or give
# something completely stupid.

import inspect
import os

import numpy as np

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from mrpy.extra.physical_dependence import mrp_b13


def test_phys():
    m = np.logspace(10,15,200)
    mrp = mrp_b13(m)
    assert mrp[-1] < 1e-20 and mrp[1] < mrp[0] < 1e-10

def test_phys_z():
    m = np.logspace(10,15,200)
    mrp1 = mrp_b13(m,z=1)
    mrp0 = mrp_b13(m,z=0)
    assert mrp1[-1] < mrp0[-1] and mrp1[0] > mrp0[0]

