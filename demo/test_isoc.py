#!/usr/bin/env python
# encoding: utf-8
"""
Test script for generating a set of isochrones and plotting them.

History
-------
2011-12-29 - Created by Jonathan Sick

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pysps import fsps
from pysps import splib

def main():
    imf, imf1, imf2, imf3, vdmc, mdave, dell, delt, sbss, fbhb, pagb = \
        0,1.3,2.3,2.3,0.08,0.5,0.,0.,0.,0.,1.
    fsps.driver.setup(1, 0)
    fsps.driver.setup_all_ssp(imf, imf1, imf2, imf3, vdmc, mdave, dell, delt,
            sbss, fbhb, pagb)
    tt = 40
    isocSet = []
    #
    for zz in range(1,23,4):
    #for zz in range(19,20):
        T, Z, isocData = compute_isochrone(tt, zz)
        isocSet.append((T, Z, isocData))
        print T, Z
    plot_isochrone_set_weights(isocSet, 5, 7, r"$J-K_s$", r"$K_s$", "isoc_test_jk",
         dist=24.5, xlim=(0.,2.),ylim=(24,12))
    plot_isochrone_set_weights(isocSet, 9,11, r"$g-i$", r"$i$", "isoc_test_gi",
            dist=24.5,xlim=(-1,5),ylim=(24,12))

def compute_isochrone(tt, zz):
    """Compute and package outputs for isochrone of age and metallicity
    indices tt and zz."""
    nBands = fsps.driver.get_n_bands()
    nMass = fsps.driver.get_n_masses_isochrone(zz, tt) # n masses for this isoc
    time, Z, massInit, logL, logT, logg, ffco, phase, wght, isocMags = \
            fsps.driver.get_isochrone(zz, tt, nMass, nBands)
    isocData = package_isochrone(massInit, logL, logT, logg, ffco, phase,
            wght, isocMags)
    return time, Z, isocData

def package_isochrone(massInit, logL, logT, logg, ffco, phase,
            wght, isocMags):
    """Repackage isochrone outputs into record arrays where the long
    dimension is mass."""
    nMass, nBands = isocMags.shape
    bandNames = []
    for bandSet in splib.FILTER_LIST:
        bandNames.append(bandSet[1])
    colDefs = np.dtype([('mass_init',np.float),('logL',np.float),
        ('logT',np.float),('logg',np.float),('ffco',np.float),
        ('phase',np.int),('wght',np.float),('mag',np.float,nBands)])
    isocData = np.empty(nMass, dtype=colDefs)
    isocData['mass_init'] = massInit
    isocData['logL'] = logL
    isocData['logT'] = logT
    isocData['logg'] = logg
    isocData['ffco'] = ffco
    isocData['phase'] = phase
    isocData['wght'] = wght
    isocData['mag'] = isocMags
    print "isocMags.shape", isocMags.shape
    print "isocData['mag'].shape", isocData['mag'].shape
    return isocData

def plot_isochrone_set(isocList, mag1Index, mag2Index, xLabel, yLabel, plotPath,
        dist=0., xlim=None, ylim=None):
    """Plot a mag1 - mag2 vs mag2 CMD for the set of isochrones."""
    Z = []
    T = []
    colour = []
    mag = []
    for zz, tt, isocData in isocList:
        Z.append(zz)
        T.append(tt)
        colour.append(isocData['mag'][:,mag1Index] - isocData['mag'][:,mag2Index])
        mag.append(isocData['mag'][:,mag2Index])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.2,0.2,0.75,0.75))
    for i in xrange(len(Z)):
        ax.plot(colour[i], mag[i]+dist, ls='-', lw=0.5, marker='o', ms=0.5) # , color='k'
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[1],ylim[0])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    fig.savefig(plotPath+".pdf", format='pdf')

def plot_isochrone_set_weights(isocList, mag1Index, mag2Index, xLabel, yLabel, plotPath,
        dist=0., xlim=None, ylim=None):
    """Plot a mag1 - mag2 vs mag2 CMD for the set of isochrones."""
    Z = []
    T = []
    colour = []
    mag = []
    weights = []
    masses = []
    phases = []
    for zz, tt, isocData in isocList:
        Z.append(zz)
        T.append(tt)
        colour.append(isocData['mag'][:,mag1Index] - isocData['mag'][:,mag2Index])
        mag.append(isocData['mag'][:,mag2Index])
        weights.append(isocData['wght'])
        masses.append(isocData['mass_init'])
        phases.append(isocData['phase'])
        print "mag length", len(isocData['mag'][:,mag2Index])
        print "masses length", len(isocData['mass_init'])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.2,0.2,0.75,0.75))
    for i in xrange(len(Z)):
        for j in xrange(len(weights[i])):
            print masses[i][j], mag[i][j]+dist, weights[i][j], phases[i][j]
        ax.plot(colour[i], mag[i]+dist, ls='-', lw=2, marker='None', ms=0., alpha=0.25) # , color='k'
        ax.scatter(colour[i], mag[i]+dist, s=10., c=weights[i], marker='o', cmap=mpl.cm.jet,
                norm=mpl.colors.Normalize(), zorder=10)
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[1],ylim[0])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    fig.savefig(plotPath+".pdf", format='pdf')

if __name__ == '__main__':
    main()


