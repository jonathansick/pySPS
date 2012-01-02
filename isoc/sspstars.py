#!/usr/bin/env python
# encoding: utf-8
"""
Generate a random distribution of stars with magnitudes derived from an
SSP isochrone and an error model for a specified set of band passes.

The intention is to provide callback functions or subclasses the provide
domain-specific error models. This base class will assume gaussian errors
with mean zero that are independent of magnitude

History
-------
2011-12-31 - Created by Jonathan Sick

"""

import numpy as np
from scipy.interpolate import interp1d

from pysps.sp_params import FILTER_LIST

def main():
    pass

class SSPStarFactory(object):
    """docstring for SSPStarFactory"""
    def __init__(self, nStars, magLims={}, massLim=0.05):
        super(SSPStarFactory, self).__init__()
        self.nStars = nStars
        self.magLims = magLims
        self.massLim = massLim
    
    def __call__(self, args):
        """Generate a random stellar catalog given an isochrone.
        
        Regarding IMF weighting
        -----------------------
        The FSPS `wght` output is an IMF weight such that at t=0 (iage=1),
        then :math:`\sum m \times w = 1`, where `m` is `mass_init`. As the
        stellar pop evolves with time, mass is lost to stellar remnants so
        that :math:`\sum m \times w < 1`.
        
        This should be kept in mind when determining the amplitudes of
        different SSPs in a CMD.
        """
        age, Z, isocData = args # outputs from SSPIsocFactory
        dM = isocData['mass_init'][1] - isocData['mass_init'][0]
        print "Total mass", isocData['mass_init'].sum()
        print "dM", dM
        print "total wght", isocData['wght'].sum()
        print "Total mass*weight", np.sum(isocData['mass_init']*isocData['wght'])
        sampleMasses = sample_isochrone_masses(isocData['mass_init'],
                isocData['wght'], self.nStars)
        print "sampleMasses:", sampleMasses
        interpMags = self.linearly_interp_mags(sampleMasses, isocData)
        # Perturb the photometry with an error model
        sampleMags = self.apply_phot_errors(interpMags) # TODO user call back?
        return sampleMags

    def linearly_interp_mags(self, sampleMasses, isocData):
        """docstring for linearly_interp_mags"""
        massGrid = isocData['mass_init']
        lowMasses = np.where(sampleMasses < massGrid[0])[0]
        highMasses = np.where(sampleMasses > massGrid[-1])[0]
        nStars = len(sampleMasses)
        cols = []
        for (bandIdx,bandName,comment) in FILTER_LIST:
            cols.append((bandName,np.float))
        sampleMags = np.empty(nStars, dtype=np.dtype(cols))
        for (bandIdx,bandName,comment) in FILTER_LIST:
            f_mag = interp1d(massGrid, isocData[bandName], bounds_error=False,
                    fill_value=np.nan)
            #interpMags = f_mag(sampleMasses)
            #print "interpMags[:,0].shape", interpMags[:,0].shape
            #print "sampleMags[%s].shape"%bandName, sampleMags[bandName].shape
            # Need to slice f_mag output to match shape of sampleMags[bandName]
            sampleMags[bandName] = f_mag(sampleMasses)#[:,0]
            sampleMags[bandName][lowMasses] = isocData[bandName][0]
            sampleMags[bandName][highMasses] = isocData[bandName][-1]
        return sampleMags

    def apply_phot_errors(self, interpMags, sigma=0.1):
        """TODO test method, need to implement user call back for errors."""
        nStars = len(interpMags)
        bandNames = interpMags.dtype.names
        randomMags = interpMags.copy()
        for bandName in bandNames:
            randomMags[bandName] += sigma*np.random.rand(nStars)
        return randomMags


def sample_isochrone_masses(massGrid, wghtGrid, nStars, minMassIndex=0):
    """Generate a random sample of `nStar` stellar masses along the isochrone.
    
    An accept-reject algorithm is used to sample masses according to
    the probability distribution function `wghtGrid`.
    """
    dM = massGrid[1] - massGrid[0]
    minMass = massGrid[minMassIndex] - dM
    maxMass = massGrid[-1] + dM
    starMasses = [] # output array of sampled stellar masses
    maxWght = wghtGrid.max()
    f_wght = interp1d(massGrid, wghtGrid)
    while len(starMasses) < nStars:
        randMass = np.random.uniform(low=minMass, high=maxMass, size=None)
        randY = np.random.uniform(low=0., high=maxWght, size=None)
        # Interpolate the weight. Special cases for the +- dM at ends of grid
        if randMass < massGrid[0]:
            w = wghtGrid[0]
        elif randMass > massGrid[1]:
            w = wghtGrid[1]
        else:
            w = f_wght(randMass)
        # Test for sample acceptance
        if randY <= w:
            # accept this sample
            starMasses.append(randMass)
    return starMasses

if __name__ == '__main__':
    main()


