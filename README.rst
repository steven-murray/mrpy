mrpy
====
.. image:: https://travis-ci.org/steven-murray/mrpy.png?branch=master
		:target: https://travis-ci.org/steven-murray/mrpy
.. image:: https://coveralls.io/repos/steven-murray/mrpy/badge.svg?branch=master&service=github
        :target: https://coveralls.io/github/steven-murray/mrpy?branch=master

A Python package for calculations with the MRP parameterisation of the Halo Mass Function.

See Murray, Robotham, Power 2016 (in prep.) for more details on what the MRP is.


Quick Start
-----------

Installation
++++++++++++
Required packages are `numpy`, `scipy`, `mpmath` (for incomplete gamma functions),
and `cached_property`.
These should be automatically installed when installing `mrpy`.

To use the MCMC fitting features, `emcee` and `pystan` are needed. These are *not*
installed automatically.

The simplest way to install is ``pip install mrpy``. This should install the required
dependencies automatically.

To get the bleeding edge, use ``pip install git+git://github.com/steven-murray/mrpy.git``.

If for some reason ``pip`` is not an option, manually download the github
repo and use ``python setup.py install``.

Getting Started
+++++++++++++++
There's a lot of things that you can do with `mrpy`. What you require will depend on the problem at hand. We recommend
looking at some of the examples, and the API itself for how to use the code.

Documentation
+++++++++++++
`Read the docs <http://mrpy.readthedocs.org>`_.


Features
--------
With `mrpy` you can:

- Calculate basic statistics of the truncated generalised gamma distribution (TGGD) with the `TGGD` class: mean,
  mode, variance, skewness, pdf, cdf, generate random variates etc.
- Generate MRP quantities with the `MRP` class: differential number counts, cumulative number counts, various methods
  for generating normalisations.
- Generate the MRP-based halo mass function as a function of physical parameters via the `mrp_b13` function.
- Fit MRP parameters to data in the form of arbitrary curves with the `get_fit_curve` function.
- Fit MRP parameters to data in the form of a sample of variates with the `PerObjFit` class: simulation data is supported
  with extra efficiency, simulation suites fitted simultaneously is also supported, arbitrary priors on parameters,
  log-normal uncertainties on variates supported through a Stan-based routine.
- Calculate analytic hessians, jacobians at any point (including the solution of a fit).
- Use alternate parameterisations of the same form via the `reparameterise` module.
- Work with a special entirely analytic model to understand the effects of various parameters in the `analytic_model` module.

Examples
--------
There are several examples featured in the ``docs/examples`` directory of the github repository. These can also be found
in the official documentation.
