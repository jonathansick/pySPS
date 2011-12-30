#!/usr/bin/env python
# encoding: utf-8
"""
Test script for generating a set of isochrones and plotting them.

History
-------
2011-12-29 - Created by Jonathan Sick

"""
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
    tt = 70
    isocSet = []
    #for zz in range(19,20):
    for zz in range(1,23,4):
        T, Z, isocData = compute_isochrone(tt, zz)
        isocSet.append((T, Z, isocData))
        print T, Z
    #plot_isochrone_set(isocSet, 5, 7, r"$J-K_s$", r"$K_s$", "isoc_test",
    #     dist=24.5, xlim=(0.,2.),ylim=(21,12))
    plot_isochrone_set(isocSet, 9,11, r"$g-i$", r"$i$", "isoc_test",)

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
        colour.append(isocData['mag'][mag1Index] - isocData['mag'][mag2Index])
        mag.append(isocData['mag'][mag2Index])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes((0.2,0.2,0.75,0.75))
    for i in xrange(len(Z)):
        ax.plot(colour[i], mag[i]+dist, ls='-', lw=0.5, marker='o', ms=0.5) # , color='k'
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1],ylim[0])
    if xlim is not None:
        #ax.set_xlim(*xlim)
        print ax.get_xlim()
    if ylim is not None:
        #ax.set_ylim(*ylim)
        print ax.get_ylim()
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    fig.savefig(plotPath+".pdf", format='pdf')


if __name__ == '__main__':
    main()


