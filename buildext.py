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
import shutil
import glob
import subprocess

fspsDir = os.environ["SPS_HOME"]

# Make FSPS
subprocess.call("cd %s;make"%os.path.join(fspsDir,"src"), shell=True)

# Gather all binaries built with FSPS
fspsSrcs = glob.glob(os.path.join(fspsDir,"src/*.o"))
oPaths = fspsSrcs
modPaths = glob.glob(os.path.join(fspsDir,"src/*.mod"))
filterNames = [os.path.join(fspsDir, 'src', n) for n in
        ['simple.o','lesssimple.o','autosps.o']]
for n in filterNames:
    oPaths = [p for p in oPaths if n not in p]

srcDir = "pysps/_fsps_src"
if not os.path.exists(srcDir): os.makedirs(srcDir)
copiedOPaths = []
newModPaths = []
for p in oPaths:
    basename = os.path.basename(p)
    shutil.copy(p, os.path.join(srcDir, basename))
for p in modPaths:
    basename = os.path.basename(p)
    newPath = os.path.join("pysps", basename)
    newModPaths.append(newPath)
    shutil.copy(p, newPath)
oPaths = glob.glob(os.path.join(srcDir, "%.o"))

subprocess.call("cd pysps;f2py fsps.f90 -m fsps -h fsps.pyf --overwrite-signature",
        shell=True)

# apparently gnu95 implies gfortran
cmd = "cd pysps;f2py -c --fcompiler=gnu95 fsps.pyf fsps.f90 %s" % " ".join(oPaths)
subprocess.call(cmd, shell=True)

# Cleanup
for p in newModPaths:
    os.remove(p)
shutil.rmtree(srcDir)
