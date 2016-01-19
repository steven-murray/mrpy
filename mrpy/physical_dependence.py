"""
Module containing functions for the dependence of MRP parameters on physical
parameters, defined with respect to Behroozi+13.

The values output by the functions in this module were derived as part of the study
reported in Murray, Robotham and Power (2016). See that paper for details on the
fits used.

In brief, three physical parameters were varied -- matter density, rms fluctuation
:math:`\sigma_8` and redshift -- around the best-fit values from Planck+13, and up
to a redshift of 8. For each, the Behroozi+13 mass function was calculated using
the `hmf` python package over a range of masses corresponding to (lm-3, lm+2), where
*lm* is the log mass mode. The MRP was fit to each curve, and then the best-fit parameters
were parameterised as a function of the physical parameters using the Eureqa software.

.. note:: Given the way these parameterisations were determined, it is not advised to
          trust these values far from the log-mass-mode of each fit.
"""
import numpy as np
import core

def _logHs_b13(z=0, Om0=0.315, sig8=0.829, mu=12.214, sd=1.6385, a=0.058562, b=1.4394,
              c=0.39111, d=0.11159, e=0.056010, f=0.42444, g=0.90369, h=0.0029417):
    """
    Return log of scale mass, Hs, as a function of physical parameters
    """
    return mu + sd*(a + b*sig8 + c*Om0 + d*sig8*z + e*z**2 + f*sig8*Om0*z - g*z - h*z**3)


def _alpha_b13(z=0, Om0=0.315, sig8=0.829, mu=-1.9097, sd=0.026906, a=2.6172, b=2.06023,
              c=1.4791, d=2.2142, e=0.53400, f=2.70981, g=0.19690):
    """
    Return power-law index, alpha, as a function of physical parameters.
    """
    return mu + sd*(a*Om0 + b*sig8 + c*d**Om0*e**z - f - g*z)


def _beta_b13(z=0, Om0=0.315, sig8=0.829, mu=0.49961, sd=0.12913, a=7.5217, b=0.18866,
             c=0.36891, d=0.071716, e=0.0029092, f=3.4453, g=0.71052):
    """
    Return cut-off parameter, beta, as a function of physical parameters.
    """
    return mu + sd*(a*sig8*Om0 - b - c*z - d*e**z - f*Om0*z*g**z)


def _lnA_b13(z=0, Om0=0.315, sig8=0.829, mu=-33.268, sd=7.3593, a=0.0029187, b=0.15541,
            c=1.4657, d=0.055025, e=0.24068, f=0.33620):
    """
    Return the natural log of the normalisation, A, in units of the pdf normalisation,
    as a function of physical parameters.
    """
    return mu + sd*(z + a*z**3 - b - c*sig8 - d*z**2 - e*sig8*z - f*Om0*z)


def mrp_params_b13(z=0, Om0=0.315, sig8=0.829, Hs_kw={}, alpha_kw={}, beta_kw={}, logA_kw={}):
    """
    Return all 4 MRP parameters as a function of physical parameters.
    """
    return _logHs_b13(z, Om0, sig8, **Hs_kw), _alpha_b13(z, Om0, sig8, **alpha_kw), \
           _beta_b13(z, Om0, sig8, **beta_kw), _lnA_b13(z, Om0, sig8, **logA_kw)


def mrp_b13(m, z=0, Om0=0.315, sig8=0.829, Hs_kw={}, alpha_kw={}, beta_kw={}, logA_kw={},
            mmin=None, norm=None, log=False, **Arhoc_kw):
    """
    Return the MRP defined at ``m`` for the given physical parameters.

    .. note :: Calls :func:`mrpy.core.mrp` in the background, and takes all of those
               parameters.
    """
    hs, alpha, beta, logA = mrp_params_b13(z, Om0, sig8, beta_kw, alpha_kw, Hs_kw, logA_kw)

    if norm is None:
        norm = np.exp(logA)

    return core.dndm(m, hs, alpha, beta, mmin, norm, log, **Arhoc_kw)
