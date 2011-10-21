Installing pySPS
================

pySPS currently lacks easy install facilities (since its in early alpha development). Nonetheless installation should be relatively easy.

Prerequisties
-------------

1. FSPS, obviously. Follow the instructions at the `FSPS homepage`_ to install a copy using SVN. Remember to set the `SPS_HOME` environment variable, pointing to the root FSPS directory on your machine.
2. Install MongoDB and have MongoDB running on your localhost. See the `MongoDB homepage`_ for more details.
3. Install the pymongo_ package, using `sudo easy_install pymongo`
4. Install PyTables_

Building pySPS
--------------

1. Copy or checkout pySPS into a directory in your python path, such as `site-packages`.
2. Copy the `src/` directory from your FSPS installation into the pySPS package directory. Re-name the directory to `fsps`.
3. Build the fortran exectuables with a `make` command. The make file relies upon your having gfortran. Adjust the Makefile as necessary for your environment. You'll want to run make clean afterwards to get rid the object files.

That's it.

.. _FSPS homepage: https://www.cfa.harvard.edu/~cconroy/FSPS.html
.. _MongoDB homepage: http://www.mongodb.org/
.. _pymongo: http://api.mongodb.org/python/current/
.. _PyTables: http://www.pytables.org
