#!/usr/bin/env python
# encoding: utf-8
"""
Manually compile the f2py extension for fsps.

.. note:: Eventually this should be converted into a setup.py function.

History
-------
2011-12-24 - Created by Jonathan Sick

"""

import os
import glob

fspsSrcs = glob.glob("fsps/*.f90")
nrSrcs = glob.glob("fsps/nr/*.f90")
fPaths = fspsSrcs + nrSrcs
oPaths = glob.glob("*.o")

# apparently gnu95 implies gfortran
cmd = "f2py -c --fcompiler=gnu95 fsps.pyf fsps.f90 %s" % " ".join(oPaths)
print cmd
os.system(cmd)

