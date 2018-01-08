Changelog
=========

Development Version
-------------------

v1.1.0 [8th Jan 2018]
---------------------
This version is the version used for all plots in Murray, Robotham, Power (2018), and is released along with that paper.
There are many changes in the code from previous versions, the result of a couple of years of sporadic work.


v1.0.0
------

Features
++++++++
- New ``PerObjFit`` class supersedes ``get_fit_perobj`` function, providing more
  coherent fitting capabilities.
- Added heaps of "real-world" examples (used in MRP paper):
    * https://github/steven-murray/mrpy/docs/examples/fit_curve_against_analytic.ipynb
    * https://github/steven-murray/mrpy/docs/examples/fit_simulation_suite.ipynb
    * https://github/steven-murray/mrpy/docs/examples/heirarchical_model_stan.ipynb
    * https://github/steven-murray/mrpy/docs/examples/explore_analytic_model.ipynb
    * https://github/steven-murray/mrpy/docs/examples/mmin_dependence.ipynb
    * https://github/steven-murray/mrpy/docs/examples/physical_dependence.ipynb
    * https://github/steven-murray/mrpy/docs/examples/parameterization_performance.ipynb
    * https://github/steven-murray/mrpy/docs/examples/SMHM.ipynb
- Added ``model`` argument to ``fit_perobj_stan`` to facilitate pickling of multiple fits.
- Added ability to send keyword arguments to priors in ``PerObjFit`` class
- Added a ``normal_prior`` function for simple normal priors.

Enhancements
++++++++++++
- Changed default weighting from 1 to 0 in ``get_fit_curve``.
- Added tests for the ``PerObjLikeWeights`` class.
- Added tests for ``nbar`` and ``rhobar`` for general ``m`` in ``MRP` subclasses.
- Changed imports so that they wouldn't show up in docs
- Many improvements to documentation (including this file!)

Bugfixes
++++++++
- Fixed issue setting ``log_mmin`` in ``IdealAnalytic``
- Fixed issue in which ``nbar`` and ``rhobar`` are wrong if ``mmin`` is not ``m.min()`` in ``MRP`` subclasses.