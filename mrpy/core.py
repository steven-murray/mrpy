"""
Basic MRP functions.
"""
import numpy as np
from special import gammainc, gamma

def ln_mrp_shape(m, hs, alpha, beta):
    r"""
    The natural log of the shape of the MRP function, without normalisation:
    :math:`\ln \left[\beta (m/H_s)^\alpha \exp(-(m/Hs)^\beta)\right].`

    Parameters
    ----------
    m : array_like
        Vector of masses at which to evaluate the MRP

    hs : float
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter

    Returns
    -------
    ln_mrp : array_like
        The log MRP values corresponding to the vector ``m``.
    """
    lny = np.log(m) - hs*np.log(10)
    return np.log(beta) + alpha * lny - np.exp(lny*beta)


def mrp_shape(m, hs, alpha, beta):
    r"""
    The shape of the MRP function, without normalisation:
    :math:`\beta (m/H_s)^\alpha \exp(-(m/Hs)^\beta).`

    Parameters
    ----------
    m : array_like
        Vector of masses at which to evaluate the MRP

    hs : float
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter

    Returns
    -------
    mrp : array_like
        The MRP values corresponding to the vector ``m``.
    """
    return  np.exp(ln_mrp_shape(m,hs,alpha,beta))

def pdf_norm(hs,alpha,beta,mmin=-np.inf,mmax=np.inf):
    """
    The normalisation, A, required to render the MRP a statistical pdf:
    :math:`A_1 = \mathcal{H}_s \Gamma(\frac{\alpha+1}{\beta},x)`.

    Parameters
    ----------
    hs : float
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter

    mmin : float, optional
        Log-10 of the lower-truncation mass. Default is zero mass.

    mmax : float, optional
        Log-10 of the upper-truncation mass. Default is infinite mass.
    """
    z = (alpha+1.0)/beta
    if np.isinf(mmin):
        lower = gamma(z)
    else:
        lower = gammainc(z,10**(beta*(mmin-hs)))

    upper = 0.0
    if not np.isinf(mmax):
        upper = gammainc(z,10**(beta*(mmax-hs)))

    return 1./(lower - upper)/10**hs

def k(hs,alpha,beta):
    r"""
    The entire integral of the un-normalised mass-weighted MRP:
    :math:`k \equiv \int_0^\infty dm m MRP(m) = \mathcal{H}_s^2 \Gamma(\frac{\alpha+2}{\beta}).`

    Parameters
    ----------
    hs : float
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter
    """
    return 10**(2*hs) * gamma((alpha+2)/beta)

def A_rhoc(hs,alpha,beta,Om0=0.3,rhoc=2.7755e11):
    r"""
    The normalisation required to bind all matter in halos of some scale:
    :math:`A_{}\rho_c} = \Omega_m \rho_c / k(\vec{\theta})`

    Parameters
    ----------
    hs : float
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter

    Om0 : float, optional
        The matter density at redshift nought.

    rhoc : float, optional
        The critical density of the Universe.
    """
    return Om0*rhoc/k(hs,alpha,beta)

def mrp(m, hs,alpha,beta,mmin=None,mmax=np.inf,norm="pdf",log=False,**Arhoc_kw):
    """
    The MRP distribution.

    Parameters
    ----------
    m : array_like
        Vector of masses at which to evaluate the MRP

    hs : float
        The base-10 logarithm of the scale mass, :math:`H_s`.

    alpha : float
        The power-law index

    beta : float
        Exponential cutoff parameter

    mmin : float, optional
        Log-10 of the lower-truncation mass. Default is the minimum mass in ``m``.

    mmax : float, optional
        Log-10 of the upper-truncation mass. Default is infinite mass.

    norm : None, string or float

    log : logical
        Whether to return the natural log of the MRP (suitable for Bayesian
        likelihoods).

    \*\*Arhoc_kw :
        Arguments directly forwarded to the mean-density normalisation.
    """
    if mmin is None:
        mmin = np.log10(m.min())


    ln_shape = ln_mrp_shape(m,hs,alpha,beta)

    if norm == "pdf":
        A = pdf_norm(hs,alpha,beta,mmin,mmax)
    elif np.isreal(norm):
        A = norm
    elif norm == "rhoc":
        A = A_rhoc(hs,alpha,beta,**Arhoc_kw)

    if log:
        return ln_shape + np.log(A)
    else:
        return np.exp(ln_shape)*A
