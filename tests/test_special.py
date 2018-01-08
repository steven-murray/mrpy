"""
Test the special functions.

Each will *at least* test each argument for float (+ve, -ve, 0), int (+ve, -ve) and array_like
input.
"""

import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)

from nose.tools import assert_raises

from mrpy.base import special as s


#===========================================================================
# Gamma()
#===========================================================================
def test_gamma_pos_float():
    assert np.isclose(s.gamma(1.2),0.9181687424)

def test_gamma_neg_float():
    assert np.isclose(s.gamma(-1.2),4.85095714052)

def test_gamma_pos_int():
    assert s.gamma(1) == 1

def test_gamma_neg_int():
    assert_raises(ValueError,s.gamma,-1)

def test_gamma_vec():
    ans = np.array([-5.738554639998505, -3.7807156398259134,
                    -3.6016112664314623,-4.680905938427317,
                    -11.923058813509279, 10.754163472232337,
                    3.386708112731563, 1.9928935227569229,
                    1.4403382059330483, 1.1642297137253033])
    assert np.all(np.isclose(s.gamma(np.linspace(-0.8,0.8,10)),ans))



#===========================================================================
# Gammainc()
#===========================================================================
def test_gammainc_pospos_float():
    assert np.isclose(s.gammainc(1.2,1.2),0.3480611830)

def test_gammainc_negpos_float():
    assert np.isclose(s.gammainc(-1.2,1.2),0.0839714)

def test_gammainc_posneg_float():
    assert_raises(TypeError,s.gammainc,1.2,-1.2)

def test_gammainc_pospos_int():
    assert np.isclose(s.gammainc(1,1),1/np.e)

def test_gammainc_negpos_int():
    assert np.isclose(s.gammainc(-1,1),0.1484955067759)

def test_gammainc_float_vec():
    ans = np.array([0.8683608186150055, 0.4956391690487021])
    assert np.all(np.isclose(s.gammainc(1.2,np.linspace(0.1,0.8,2)),ans))

def test_gammainc_vec_float():
    ans = np.array([0.10238660549891246, 0.262160799331728])
    assert np.all(np.isclose(s.gammainc(np.array([-0.8,0.8]),1.2),ans))

def test_gammainc_vec_vec():
    ans = np.array([5.277378904974033, 0.41093548285294645])
    assert np.all(np.isclose(s.gammainc(np.array([-0.8,0.8]),np.array([0.1,0.8])),ans))



#===========================================================================
# Polygamma()
#===========================================================================
def test_polygamma_pos():
    assert np.isclose(s.polygamma(1,1.2),1.26738)

def test_polygamma_neg():
    assert np.isclose(s.polygamma(1,-1.2),27.9939)

def test_polygamma_vec():
    ans = np.array([-10.4238, -0.577216 ])
    assert np.all(np.isclose(s.polygamma(0,np.array([0.1,1.0])),ans))



#===========================================================================
# G1() has gammainc() embedded
#===========================================================================
# def test_G1_pospos_float():
#     assert np.isclose(s.G1(1.2,1.2),0.555948675025 * s.gammainc(1.2,1.2))
#
# def test_G1_negpos_float():
#     assert np.isclose(s.G1(-1.2,1.2),0.299019047207 * s.gammainc(-1.2,1.2))
#
# def test_G1_posneg_float():
#     assert_raises(TypeError,s.G1,1.2,-1.2)
#
# def test_G1_pospos_int():
#     assert np.isclose(s.G1(1,1),0.596347362323 * s.gammainc(1,1))
#
# def test_G1_negpos_int():
#     assert np.isclose(s.G1(-1,1),0.341103314565 * s.gammainc(-1,1))
#
# def test_G1_float_vec():
#     ans = np.array([ 2.17803164,  0.73639127]) * s.gammainc(1.2,np.linspace(0.1,0.8,2))
#     assert np.all(np.isclose(s.G1(1.2,np.linspace(0.1,0.8,2)),ans))
#
# def test_G1_vec_float():
#     ans = np.array([ 0.32843287,  0.49768655]) * s.gammainc(1.2,np.linspace(0.1,0.8,2))
#     assert np.all(np.isclose(s.G1(np.array([-0.8,0.8]),1.2),ans))
#
# def test_G1_vec_vec():
#     ans = np.array([ 0.7714453,   0.64866584])
#     assert np.all(np.isclose(s.G1(np.array([-0.8,0.8]),np.array([0.1,0.8])),ans))




#===========================================================================
# G2() has gammainc() embedded
#===========================================================================
# def test_G2_pospos_float():
#     assert np.isclose(s.G2(1.2,1.2),0.23180020301)
#
# def test_G2_negpos_float():
#     assert np.isclose(s.G2(-1.2,1.2),0.0789290760128)
#
# def test_G2_posneg_float():
#     assert_raises(TypeError,s.G2,1.2,-1.2)
#
# def test_G2_pospos_int():
#     assert np.isclose(s.G2(1,1),0.265965385032)
#
# def test_G2_negpos_int():
#     assert np.isclose(s.G2(-1,1),0.101341905607)
#
# def test_G2_float_vec():
#     ans = np.array([ 2.77211686,  0.38713129])
#     assert np.all(np.isclose(s.G2(1.2,np.linspace(0.1,0.8,2)),ans))
#
# def test_G2_vec_float():
#     ans = np.array([ 0.09337627,  0.19228424])
#     assert np.all(np.isclose(s.G2(np.array([-0.8,0.8]),1.2),ans))
#
# def test_G2_vec_vec():
#     ans = np.array([ 0.50623584,  0.31357118])
#     assert np.all(np.isclose(s.G2(np.array([-0.8,0.8]),np.array([0.1,0.8])),ans))
#




#===========================================================================
# hyperreg() has gamma() embedded
#===========================================================================
def test_hyperReg_2F2_pospos_float():
    assert np.isclose(s.hyperReg_2F2(1.2,1.2),0.596919004394)

def test_hyperReg_2F2_negpos_float():
    assert np.isclose(s.hyperReg_2F2(-1.2,1.2),-1.20091607472)

def test_hyperReg_2F2_posneg_float():
    assert np.isclose(s.hyperReg_2F2(1.2,-1.2),1.22515135239)

def test_hyperReg_2F2_pospos_int():
    assert np.isclose(s.hyperReg_2F2(1,1),0.796599599297)

def test_hyperReg_2F2_negpos_int():
    assert_raises(ZeroDivisionError,s.hyperReg_2F2,-1,1)

def test_hyperReg_2F2_posneg_int():
    assert np.isclose(s.hyperReg_2F2(1,-1),1.31790215145)

def test_hyperReg_2F2_float_vec():
    ans = np.array([ 0.79980452,  0.65968203])
    assert np.all(np.isclose(s.hyperReg_2F2(1.2,np.linspace(0.1,0.8,2)),ans))

def test_hyperReg_2F2_vec_float():
    ans = np.array([-0.84994678,  0.93493122])
    assert np.all(np.isclose(s.hyperReg_2F2(np.array([-0.8,0.8]),1.2),ans))

def test_hyperReg_2F2_vec_vec():
    ans = np.array([-0.0283642,  0.99684913])
    assert np.all(np.isclose(s.hyperReg_2F2(np.array([-0.8,0.8]),np.array([0.1,0.8])),ans))

