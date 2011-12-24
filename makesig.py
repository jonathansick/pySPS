#!/usr/bin/env python
# encoding: utf-8
import os

os.system("f2py fsps.f90 -m fsps -h fsps.pyf --overwrite-signature")
