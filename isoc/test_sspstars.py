#!/usr/bin/env python
# encoding: utf-8
"""
Test module for SSPStarFactory and SSPIsocFactory

History
-------
2011-12-31 - Created by Jonathan Sick

"""

import numpy as np

from pysps import sp_params
from sspstars import SSPStarFactory
from sspisoc import SSPIsocFactory

def main():
    pset = sp_params.ParameterSet(None, zmet=19)
    pset.p['iage'] = 1 #70 # manually insert an age index
    isocFactory = SSPIsocFactory(vega_mags=0)
    age, Z, isocData = isocFactory(pset)
    print age, Z
    starFactory = SSPStarFactory(10000)
    starFactory((age, Z, isocData))

if __name__ == '__main__':
    main()


