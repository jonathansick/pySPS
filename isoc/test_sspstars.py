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
    pset.p['iage'] = 0 #70 # manually insert an age index
    isocFactory = SSPIsocFactory(vega_mags=0)
    age, Z, isocData = isocFactory(pset)
    print age, Z
    plot_phase_imf(isocData, "phase_imf")
    #starFactory = SSPStarFactory(10000, massLim=1.)
    #mags = starFactory((age, Z, isocData))

    #fig = plt.figure(figsize=(6,6))
    #ax = fig.add_axes((0.2,0.2,0.75,0.75))
    #ax.scatter(mags['MegaCam_u']-mags['MegaCam_g'], mags['MegaCam_g'],
    #        marker='o', s=2, alpha=0.25, edgecolor='none', facecolor='k')
    #ylim = ax.get_ylim()
    #ax.set_ylim(ylim[1],ylim[0])
    #fig.savefig("synth_cmd.png", format="png", dpi=300)

def plot_phase_imf(isocData, plotPath):
    """docstring for plot_phase_imf"""
    M = isocData['mass_init']
    wght = isocData['wght']
    phases = isocData['phase']
    phaseSet = list(set(phases.tolist()))
    phaseSet.sort()

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes((0.15,0.15,0.8,0.8))
    for p in phaseSet:
        inPhase = np.where(phases == p)[0]
        ax.semilogy(M[inPhase], M[inPhase]*wght[inPhase], '-o')
    fig.savefig(plotPath+".pdf", format="pdf")



if __name__ == '__main__':
    main()


