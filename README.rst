mrpy
====

A Python package for calculations with the MRP parameterisation of the Halo Mass Function.

Installation
------------
Required packages are numpy, scipy and mpmath (for incomplete gamma functions).

Once these are installed (``"pip install numpy scipy mpmath"``), just navigate to
the directory in which this file is kept, and type ``python setup.py install``.

It is possible that the dependencies will automatically install anyway.

Basic Usage
-----------
Currently, only very limited functionality is implemented. All basic MRP functions
are in the ``core`` module. Generally, one will want to do:

    >>> from mrpy.core import mrp
    >>> import numpy as np
    >>> m = np.logspace(10,15,500)
    >>> dndm = mrp(m,<hs>,<alpha>,<beta>,...)

Please look at the docstring of ``mrp`` for more details of options (eg. it can
be normalised in a few different ways or returned as the log).

Additionally, ``physical_dependence`` contains a couterpart to this function,
``mrp_b13``, which returns the MRP at a given mass vector, according to input
physical variables, $z$, $\Omega_m$ and $\sigma_8$. 
