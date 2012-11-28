pySPS
=====

pySPS is a python interface to the [Flexible Stellar Population Synthesis](https://www.cfa.harvard.edu/~cconroy/FSPS.html)
package by Conroy, Gunn and White.

Note this package is still in early development! Use with caution.

Installation
------------

pySPS requires FSPS to be installed separately. Some modules will require pymongo and pytables. pySPS itself can be installed by running:

    python setup.py build_fsps
    python setup.py develop

You should then be able to `import pysps`

Documentation
-------------

[Sphinx](http://sphinx.pocoo.org/) documentation can be built via

    python setup.py make_sphinx

and HTML documentation is readable from `build/sphinx/html`.


Contact
-------

Contact Jonathan Sick (jsick@astro.queensu.ca) for questions and issues related to pySPS. For questions about FSPS, contact Charlie Conroy. This package is not endorsed by the FSPS authors.

License
-------

Copyright (c) 2011-2012 Jonathan Sick
All rights reserved.

Consult LICENSE.rst for licensing details.
