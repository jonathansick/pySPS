.. pySPS documentation master file, created by
   sphinx-quickstart on Tue Oct  4 21:34:51 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pySPS - Production of Stellar Population Libraries in Python
============================================================

pySPS is a python wrapper around the `Flexible Stellar Population Synthesis`_ code, by Conroy, Gunn and White (2009) [1]_ [2]_. pySPS can be used to specify and create grids of stellar populations.

An emphasis is put on parallel processing, and handling of large tables. For example MongoDB_ is used to store model specifications and capture output from distributed processes. HDF5 tables allow observations to fit fitted over output grids that may not fit in a machine's memory.

Contents:

.. toctree::
   :maxdepth: 2

   fsps
   filters

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. [1] Conroy, C., Gunn, J. and White, M. "The Propagation of Uncertainties in Stellar Population Synthesis Modeling. I. The Relevance of Uncertain Aspects of Stellar Evolution and the Initial Mass Function to the Derived Physical Properties of Galaxies," `ApJ 699 pp. 486-506 <http://adsabs.harvard.edu/abs/2009ApJ...699..486C>`_, 2009.
.. [2] Conroy, C. and Gunn, J. "The Propagation of Uncertainties in Stellar Population Synthesis Modeling. III. Model Calibration, Comparison, and Evaluation," `ApJ 712 pp. 833-857 <http://adsabs.harvard.edu/abs/2010ApJ...712..833C>`_, 2010.

.. _Flexible Stellar Population Synthesis: https://www.cfa.harvard.edu/~cconroy/FSPS.html
.. _MongoDB: http://www.mongodb.org/
