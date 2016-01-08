mrpy
====

A Python package for calculations with the MRP parameterisation of the Halo Mass Function.

See Murray, Robotham, Power 2016 (in prep.) for more details on what the MRP is.

.. image:: https://travis-ci.org/steven-murray/mrpy.png?branch=master
		:target: https://travis-ci.org/steven-murray/mrpy
.. image:: https://coveralls.io/repos/steven-murray/mrpy/badge.svg?branch=master&service=github
        :target: https://coveralls.io/github/steven-murray/mrpy?branch=master

Quick Start
-----------

Installation
++++++++++++
Required packages are `numpy`, `scipy`, `mpmath` (for incomplete gamma functions),
and `cached_property`.
These should be automatically installed when installing `mrpy`.

To install, simply use ``pip install git+git://github.com/steven-murray/mrpy.git``.
This should install all dependencies and the `mrpy` package.

If this is not an option, manually download the github repo and use ``python setup.py install``.
In this case, you may need to manually install the dependencies first.

Core Functionality
++++++++++++++++++
Core functionality (i.e. calculation of the MRP function given input parameters,
plus some other functions useful for normalising) is in the ``core`` module. As
an example::

    >>> from mrpy import dndm
    >>> import numpy as np
    >>> m = np.logspace(10,15,500)
    >>> dn = dndm(m,<hs>,<alpha>,<beta>,...)

Please look at the docstring of ``dndm`` for more details of options (eg. it can
be normalised in a few different ways or returned as the log).

Pure Stats
++++++++++
If you don't care so much about the fact that the MRP is good for halo mass functions
(or don't know what a halo mass function is...), but want to use the statistical
distribution, you'll want the ``stats`` module. It contains an object called ``TGGD``
(short for Truncated Generalised Gamma Distribution), which has many statistical
quantities and methods available (such as producing random variates, mean, mode etc.)

Physical dependence
+++++++++++++++++++
The ``physical_dependence`` module contains a counterpart to the basic ``dndm``
function, called ``mrp_b13``, which returns the best-fit MRP according to input
physical variables (redshift, matter density, rms mass variance). These are
derived from fits to the theoretical mass function of Behroozi+2013. Example::

    >>> from mrpy.physical_dependence import mrp_b13
    >>> import numpy as np
    >>> m = np.logspace(10,16,600)
    >>> dndm_z1 = mrp_b13(m,z=1)
    >>> dndm_z0 = mrp_b13(m,sigma_8=0.85)

Note that the default redshift and cosmology is z=0, sigma_8=0.829 and Om0 = 0.315,
in accordance with Planck+13 results.

Fitting MRP
+++++++++++
The ``fit_curve`` module contains routines to fit the MRP to binned/curve data.
This can be a theoretical curve, or binned halos (or other variates). There are
several options available, and the gradient of the objective function is specified analytically
to improve performance. See Murray, Robotham, Power, 2016 (in prep.) for more details.

An example::

    >>> import mrpy
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> lm = np.logspace(10,15,500)
    >>> dndm = mrpy.dndm(m,hs=14.0,alpha=-1.9,beta=0.75,norm=1)
    >>> result, curve_obj = mrpy.get_fit_curve(logm,dndm,hs0=14.1,alpha0=-1.8,beta0=0.81,lnA0=1)
    >>> print result
    >>> plt.plot(curve_obj.logm,curve_obj.dndm()/dndm-1)
    >>> plt.xscale('log')

This simply fits the four MRP parameters to the input curve. Options can be
specified to constrain the normalisation either via the integral of the data, or the known mean density
of the Universe (or some combination thereof).

To fit actual samples of halos, use the ``fit_perobj`` module. There are three ways
to do the fitting in this module. The first is to use simple downhill-gradient methods.
Find an example of that below. The other methods both use MCMC. One uses the ``emcee``
package, and is overall more flexible but does not support arbitrary per-object uncertainties.
The other uses the ``pystan`` package, and is less flexible, but can take arbitrary uncertainties.

An example of using the downhill method::

    >>> from mrpy.stats import TGGD
    >>> r = TGGD(scale=1e14,a=-1.8,b=1.0,xmin=1e12).rvs(1e5)
    >>> from mrpy.fit_perobj import fit_perobj_opt
    >>> res,obj = fit_perobj_opt(r)
    >>> print res.x
    >>> print obj.hessian
    >>> print obj.cov
    >>> from matplotlib.pyplot import plot
    >>> plot(obj.logm,obj.dndm(log=True))
    >>> print obj.stats.mean, r.mean()


These have been a list of possibly the more commonly required methods. There's more inside!
