mrpy
====
.. image:: https://travis-ci.org/steven-murray/mrpy.png?branch=master
		:target: https://travis-ci.org/steven-murray/mrpy
.. image:: https://coveralls.io/repos/steven-murray/mrpy/badge.svg?branch=master&service=github
        :target: https://coveralls.io/github/steven-murray/mrpy?branch=master
.. image:: https://api.codacy.com/project/badge/Grade/e5d5d9b72d024bd09e24ea833745c6da
        :target: https://www.codacy.com/app/steven-murray/mrpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=steven-murray/mrpy&amp;utm_campaign=Badge_Grade

A Python package for calculations with the MRP parameterisation of the Halo Mass Function.

See `Murray, Robotham, Power (2018) <http://arxiv.org/abs/1801.02723>`_ for more details on what the MRP is.


Documentation
+++++++++++++
`Read the docs <http://mrpy.readthedocs.org>`_.


Quick Start
-----------

Installation
++++++++++++
``>> pip install mrpy``.

This should install the required dependencies automatically.

Note, to use the MCMC fitting features, `emcee` is needed. This is *not* installed automatically.

To get the bleeding edge, use ``pip install git+git://github.com/steven-murray/mrpy.git``.

Getting Started
+++++++++++++++
There's a lot of things that you can do with `mrpy`. What you require will depend on the problem at hand. We recommend
looking at some of the examples, and the API itself for how to use the code.

Features
--------
With `mrpy` you can:

- Calculate basic statistics of the truncated generalised gamma distribution (TGGD) with the `TGGD` class: mean,
  mode, variance, skewness, pdf, cdf, generate random variates etc.
- Generate MRP quantities with the `MRP` class: differential number counts, cumulative number counts, various methods
  for generating normalisations.
- Generate the MRP-based halo mass function as a function of physical parameters via the `mrp_b13` function.
- Fit MRP parameters to data in the form of arbitrary curves with the `get_fit_curve` function.
- Fit MRP parameters to data in the form of a sample of variates with the `SimFit` class: simulation data is supported
  with extra efficiency, simulation suites fitted simultaneously is also supported, arbitrary priors on parameters,
  log-normal uncertainties on variates supported.
- Calculate analytic hessians, jacobians at any point (including the solution of a fit).
- Use alternate parameterisations of the same form via the `reparameterise` module.
- Work with a special entirely analytic model to understand the effects of various parameters in the `analytic_model` module.

Examples
--------
There are several examples featured in the ``docs/examples`` directory of the github repository. These can also be found
in the official documentation.

Acknowledging
-------------
If you use this code in your work, please cite `Murray, Robotham, Power (2018) <http://arxiv.org/abs/1801.02723>`_.
Also consider starring/following the repo on github so we know how much it is being used.
We would also love any input to the code!
