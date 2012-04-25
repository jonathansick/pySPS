#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup
from sphinx.setup_command import BuildDoc

dependencies = """
sphinx
numpy
pymongo
"""

cmdclass = {'build_sphinx': BuildDoc}

setup(
    name='pysps',
    version=0.2,
    author='Jonathan Sick',
    author_email='jonathansick@mac.com',
    description='Interface to FSPS, the Flexible Stellar Population Synthesis package',
    license='BSD',
    install_requires=dependencies.split(),
    cmdclass=cmdclass
)
