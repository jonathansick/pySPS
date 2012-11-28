#!/usr/bin/env python
# encoding: utf-8

import os
import shutil
import glob
import subprocess
from setuptools import setup, Command
from sphinx.setup_command import BuildDoc

desc = open("README.md").read()
dependencies = """
sphinx
numpy
pymongo
"""


class BuildFSPS(Command):
    """setuptools command for compiling the fsps extension."""
    description = "Build FSPS and the fsps extension module"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """docstring for run"""
        fspsDir = os.environ["SPS_HOME"]

        # Make FSPS
        subprocess.call("cd %s;make" % os.path.join(fspsDir, "src"),
                shell=True)

        # Gather all binaries built with FSPS
        fspsSrcs = glob.glob(os.path.join(fspsDir, "src/*.o"))
        oPaths = fspsSrcs
        modPaths = glob.glob(os.path.join(fspsDir, "src/*.mod"))
        # Omit the objects of the command line programs
        filterNames = [os.path.join(fspsDir, 'src', n) for n in
                ['simple.o', 'lesssimple.o', 'autosps.o']]
        for n in filterNames:
            oPaths = [p for p in oPaths if n not in p]

        srcDir = "pysps/_fsps_src"
        if not os.path.exists(srcDir):
            os.makedirs(srcDir)
        newModPaths = []
        for p in oPaths:
            basename = os.path.basename(p)
            shutil.copy(p, os.path.join(srcDir, basename))
        for p in modPaths:
            basename = os.path.basename(p)
            newPath = os.path.join("pysps", basename)
            newModPaths.append(newPath)
            shutil.copy(p, newPath)
        copiedoPaths = glob.glob(os.path.join(srcDir, "%.o"))

        # Build the f2py signature file
        subprocess.call(
            "cd pysps;f2py fsps.f90 -m fsps -h fsps.pyf --overwrite-signature",
            shell=True)

        # Compile the f2py module
        # apparently gnu95 implies gfortran
        cmd = "cd pysps;f2py -c --fcompiler=gnu95 fsps.pyf fsps.f90 %s" \
                % " ".join(copiedoPaths)
        subprocess.call(cmd, shell=True)

        # Cleanup
        for p in newModPaths:
            os.remove(p)
        shutil.rmtree(srcDir)


cmdclass = {'build_sphinx': BuildDoc, 'build_fsps': BuildFSPS}

setup(
    name='pysps',
    version=0.2,
    author='Jonathan Sick',
    author_email='jonathansick@mac.com',
    url='https://github.com/jonathansick/pySPS',
    description='Interface to Flexible Stellar Population Synthesis package',
    license='BSD',
    install_requires=dependencies.split(),
    cmdclass=cmdclass,
    long_description=desc,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
