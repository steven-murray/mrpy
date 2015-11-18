mrpy
====

A Python package for calculations with the MRP parameterisation of the Halo Mass Function.

.. image:: https://travis-ci.org/steven-murray/mrpy.png?branch=master
		:target: https://travis-ci.org/steven-murray/mrpy
.. image:: https://pypip.in/d/mrpy/badge.png
        :target: https://pypi.python.org/pypi/mrpy/
.. image:: https://pypip.in/v/mrpy/badge.png
        :target: https://pypi.python.org/pypi/mrpy/

Installation
------------
Required packages are numpy, scipy and mpmath (for incomplete gamma functions).
These should be automatically installed when installing ``mrpy``.

To install, simply use ``pip install git+git://github.com/steven-murray/mrpy.git``.

Usage
-----
Currently, limited functionality is implemented.

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
function, called ``mrp_b13``, which returns the MRP according to input
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

Utilities
+++++++++
Utility functions are located in the ``utilities`` module, and at the moment
comprises just one function for generating samples from the MRP distribution,
called as::

    >>> from mrpy.utilities import generate_masses
    >>> sample = generate_masses(N=14000,hs=14.0,alpha=-1.9,beta=0.8,mmin=10)

Coming Soon
-----------
Upcoming features include:

* Proper calculation of Hessians when fitting MRP to curves
* Ability to fit robustly to samples of halos using downhill gradient methods.
* Ability to use MCMC methods to fit to samples of halos
* STAN modules for MCMC fits to halo samples with arbitrary mass uncertainty
* Methods for defining Stellar-mass Halo-Mass relations
* Implementation of a fast, analytic, ideal likelihood based on fitting to samples of halos drawn from a pure MRP distribution.
* A useful re-parameterisation of the MRP
* Online docs
