Changelog
=========

Development Version
-------------------

Features
++++++++
- New ``PerObjFit`` class supersedes ``get_fit_perobj`` function, providing more
  coherent fitting capabilities.
- Added heaps of "real-world" examples (used in MRP paper):
    - explore_analytic_model
    - fit_curve_against_analytic
    - fit_simulation_suite
    - heirarchical_model_stan
    - mmin_dependence
    - physical_dependence
    - reparameterization_performance
- Added ``model`` argument to ``fit_perobj_stan`` to facilitate pickling of multiple fits.
- Added ability to send keyword arguments to priors in ``PerObjFit`` class
- Added a ``normal_prior`` function for simple normal priors.

Enhancements
++++++++++++
- changed default weighting from 1 to 0 in ``get_fit_curve``.
- added tests for the ``PerObjLikeWeights`` class.
- added tests for ``nbar`` and ``rhobar`` for general ``m`` in ``MRP` subclasses.
- changed imports so that they wouldn't show up in docs

Bugfixes
++++++++
- Fixed issue setting ``log_mmin`` in ``IdealAnalytic``
- Fixed issue in which ``nbar`` and ``rhobar`` are wrong if ``mmin`` is not ``m.min()`` in ``MRP`` subclasses.