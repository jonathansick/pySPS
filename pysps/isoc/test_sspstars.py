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
    pset.p['iage'] = 70 #70 # manually insert an age index
    isocFactory = SSPIsocFactory(vega_mags=0)
    age, Z, isocData = isocFactory(pset)
    print age, Z
    plot_phase_imf(isocData, "phase_imf")
    test_mass_monotonicity(isocData)
    starFactory = SSPStarFactory(100000, massLim=2.1)
    mags = starFactory((age, Z, isocData))

    plot_cmd_phase_isoc(mags, isocData)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.2,0.2,0.75,0.75))
    ax.scatter(mags['MegaCam_u']-mags['MegaCam_g'], mags['MegaCam_g'],
            marker='o', s=2, alpha=0.25, edgecolor='none', facecolor='k')
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1],ylim[0])
    fig.savefig("synth_cmd.png", format="png", dpi=300)

def plot_cmd_phase(stars):
    """docstring for plot_cmd_phase"""
    phases = list(set(stars['phase'].tolist()))
    phases.sort()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.15,0.15,0.75,0.75))
    colours = ['k','r','b','c','m','y']
    for p, c in zip(phases, colours):
        pp = np.where(stars['phase'] == p)[0]
        print "%i stars of type %i" % (len(pp),p)
        ax.scatter(stars['MegaCam_g'][pp]-stars['MegaCam_i'][pp],
                stars['MegaCam_i'][pp],
                marker='o', s=2, alpha=1., edgecolor='none',facecolor=c,
                zorder=p)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1],ylim[0])
    ax.set_xlabel(r"$g^\prime - i^\prime$")
    ax.set_ylabel(r"$i^\prime$")
    fig.savefig("synth_cmd_phases.png", format="png", dpi=300)


def plot_cmd_phase_isoc(stars, isocData):
    """docstring for plot_cmd_phase"""
    phases = list(set(isocData['phase'].tolist()))
    phases.sort()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.15,0.15,0.75,0.75))
    ax.scatter(stars['MegaCam_g']-stars['MegaCam_i'],
                stars['MegaCam_i'],
                marker='o', s=2, alpha=1., edgecolor='none',facecolor='0.5',
                zorder=0)
    # Overplot isochrone, coloured by phase
    colours = ['r','g','b','c','m','y']
    for p, c in zip(phases, colours):
        pp = np.where(isocData['phase'] == p)[0]
        ax.plot(isocData['MegaCam_g'][pp]-isocData['MegaCam_i'][pp],
            isocData['MegaCam_i'][pp], ls='-', c=c, lw=2, alpha=0.5,
            marker='None')
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1],ylim[0])
    ax.set_xlabel(r"$g^\prime - i^\prime$")
    ax.set_ylabel(r"$i^\prime$")
    fig.savefig("synth_cmd_phases_isoc.png", format="png", dpi=300)

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

#test_mass_monotonicity(isocData)
def test_mass_monotonicity(isocData):
    """Verfy that the mass of each successive index is greater than the prev."""
    massArray = isocData['mass_init']
    for i in xrange(1, len(massArray)):
        if massArray[i] <= massArray[i-1]:
            print "Index %i not monotonic",
            print "phases:", isocData['phase'][i], isocData['phase'][i-1]
    print "Finished monotonicity check"



if __name__ == '__main__':
    main()


