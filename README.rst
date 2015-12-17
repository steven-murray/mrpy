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

If this is not an option, manually download this repo and use ``python setup.py install``.
In this case, you may need to manually install the dependencies first.

Core Functionality
++++++++++++++++++

Core functionality (i.e. calculation of the MRP function given input parameters,
plus some other functions useful for normalising) is in the ``core`` module. As
an example::

    >>> from mrpy import mrp
    >>> import numpy as np
    >>> m = np.logspace(10,15,500)
    >>> dndm = mrp(m,<hs>,<alpha>,<beta>,...)

Please look at the docstring of ``mrp`` for more details of options (eg. it can
be normalised in a few different ways or returned as the log).

Physical dependence
+++++++++++++++++++
The ``physical_dependence`` module contains a counterpart to the basic ``mrp``
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

Simple Fits
+++++++++++
The ``simple_fits`` module contains routines to fit the MRP to binned/curve data.
This can be a theoretical curve, or binned halos. There are several options
available, and the gradient of the objective function is specified analytically
to improve performance. See Murray et al. (in prep.) for more details.

An example::

    >>> import mrpy
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> lm = np.logspace(10,15,500)
    >>> dndm = mrpy.mrp(m,hs=14.0,alpha=-1.9,beta=0.75,norm=1)
    >>> result, fitted_curve = mrpy.get_fit_curve(logm,dndm,hs0=14.1,alpha0=-1.8,beta0=0.81,lnA0=1)
    >>> print result
    >>> plt.plot(m,fitted_curve/dndm-1)
    >>> plt.xscale('log')

This simply fits the four MRP parameters to the input curve (which itself is a
perfect MRP curve with known parameters). Options can be specified to constrain
the normalisation either via the integral of the data, or the known mean density
of the Universe (or some combination thereof).

Coming Soon
-----------
Upcoming features include:

* Ability to fit robustly to samples of halos using downhill gradient methods.
* Ability to use MCMC methods to fit to samples of halos
* STAN modules for MCMC fits to halo samples with arbitrary mass uncertainty
* A useful re-parameterisation of the MRP
