#!/usr/bin/env python
# encoding: utf-8
"""
Test module for SSPStarFactory and SSPIsocFactory

History
-------
2011-12-31 - Created by Jonathan Sick

"""

import numpy as np
import matplotlib.pyplot as plt

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
    mags = starFactory((age, Z, isocData))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.2,0.2,0.75,0.75))
    ax.scatter(mags['MegaCam_u']-mags['MegaCam_g'], mags['MegaCam_g'],
            marker='o', s=0.5, alpha=0.5, edgecolor='none', facecolor='k')
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1],ylim[0])
    fig.savefig("synth_cmd.png", format="png", dpi=300)



if __name__ == '__main__':
    main()


