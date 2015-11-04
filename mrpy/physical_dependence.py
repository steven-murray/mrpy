"""
Module containing functions for the dependence of MRP parameters on physical
parameters, defined with respect to Behroozi+13.
"""
import numpy as np
from core import mrp, pdf_norm

def Hs_b13(z=0,Om0=0.315,sig8=0.829,mu=12.216,sd=1.6413,a=1.6955,b=5.4038,
           c=0.88450,d=5.7651):
    """
    Return log of scale mass, Hs, as a function of physical parameters
    """
    return  mu+ sd*(Om0 + a*sig8 + b*c**z - d)


def alpha_b13(z=0,Om0=0.315,sig8=0.829,mu=-1.9100,sd=0.026819,a=3.0460,b=0.17600,
              c=1.8861,d=1.5235):
    """
    Return power-law index, alpha, as a function of physical parameters.
    """
    return mu + sd*(a*Om0 + sig8**2 + b*Om0*z**(-c*z) - d*np.log(1.0 + z))

def beta_b13(z=0,Om0=0.315,sig8=0.829,mu=0.50056,sd=0.12893,a=6.2701,b=2.0153,
             c=0.53101,d=1.8976,e=0.56778):
    """
    Return cut-off parameter, beta, as a function of physical parameters.
    """
    return mu + sd*(a*sig8*Om0 + b*c**z - d - e*Om0*z)

def logA_b13(z=0,Om0=0.315,sig8=0.829,mu=3.9293,sd=4.6086,a=1.0587,b=0.0029051,
             c=0.25536,d=1.3090,e=0.049780,f=0.32113,g=0.42027):
    """
    Return the natural log of the normalisation, A, in units of the pdf normalisation,
    as a function of physical parameters.
    """
    return mu + sd*(a*z + b*z**3 - c - d*sig8 - e*z**2 - f*sig8*z - g*Om0*z)

def mrp_params_b13(z=0,Om0=0.315,sig8=0.829,Hs_kw={},alpha_kw={},beta_kw={},logA_kw={}):
    """
    Return all 4 MRP parameters as a function of physical parameters.
    """
    return Hs_b13(z,Om0,sig8,**Hs_kw), alpha_b13(z,Om0,sig8,**alpha_kw), \
           beta_b13(z,Om0,sig8,**beta_kw), logA_b13(z,Om0,sig8,**logA_kw)

def mrp_b13(m,z=0,Om0=0.315,sig8=0.829,Hs_kw={},alpha_kw={},beta_kw={},logA_kw={},
            mmin=None,mmax=np.inf,norm=None,log=False,**Arhoc_kw):
    """
    Return the MRP defined at ``m`` for the given physical parameters.

    .. note :: Calls :func:`core.mrp` in the background, and takes all of those
               parameters.
    """
    hs, alpha,beta,logA = mrp_params_b13(z,Om0,sig8,beta_kw,alpha_kw,Hs_kw,logA_kw)

    if norm is None:
        if mmin is None:
            mmin = np.log10(m.min())
        norm = np.exp(logA) * pdf_norm(hs,alpha,beta,mmin,mmax)

    return mrp(m,hs,alpha,beta,mmin,mmax,norm,log,**Arhoc_kw)
