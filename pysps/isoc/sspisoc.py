#!/usr/bin/env python
# encoding: utf-8
"""
Generate a single isochrone for a given age and metallicity.

History
-------
2011-12-31 - Created by Jonathan Sick

"""
import numpy as np
from pysps import fsps
from pysps.sp_params import FILTER_LIST

def main():
    pass

class SSPIsocFactory(object):
    """Generate single isochrones (single age and metallicity)."""
    def __init__(self, vega_mags=0):
        super(SSPIsocFactory, self).__init__()
        self.compute_vega_mags = vega_mags
        self.fspsInit = False

    def __call__(self, p):
        """Generate an isochrone.

        Parameters
        ----------
        p : dict
            Dictionary with FSPS parameters and *index* of the isochrone age
            to be returned. Required keys are: imf_type, imf1, imf2, imf3, vdmc, &
            mdave, dell, delt, sbss, fbhb, pagb *and* iage, the index of isoc
            age starting at 1, *and* `zmet`, the metallicity index starting at 1
        """
        if self.fspsInit == False:
            fsps.driver.setup(self.compute_vega_mags, 0)
        # TODO eventually want to cache whether we really need to regenerate
        # an SSP or not based on previous computations
        fsps.driver.setup_one_ssp(p['zmet'], p['imf_type'], p['imf1'],
                p['imf2'], p['imf3'], p['vdmc'], p['mdave'], p['dell'],
                p['delt'], p['sbss'], p['fbhb'], p['pagb'])
        nAges = fsps.driver.get_n_ages_isochrone()
        nBands = fsps.driver.get_n_bands()
        nMasses = fsps.driver.get_n_masses_isochrone(p['zmet'], p['iage'])
        age, Z, massInit, logL, logT, logg, ffco, phase, wght, mags \
                = fsps.driver.get_isochrone(p['zmet'], p['iage'], nMasses, nBands)
        isocData = self.package_isochrone(massInit, logL, logT, logg, ffco,
                phase, wght, mags)
        return age, Z, isocData

    def package_isochrone(self, massInit, logL, logT, logg, ffco, phase,
            wght, isocMags):
        """Repackage isochrone outputs into record arrays where the long
        dimension is mass."""
        nMass, nBands = isocMags.shape
        bandNames = []
        for bandSet in FILTER_LIST:
            bandNames.append(bandSet[1])
        cols = [('mass_init',np.float),('logL',np.float),
            ('logT',np.float),('logg',np.float),('ffco',np.float),
            ('phase',np.int),('wght',np.float)]
        for (bandIdx,bandName,comment) in FILTER_LIST:
            cols.append((bandName,np.float))
        colDefs = np.dtype(cols)
        isocData = np.empty(nMass, dtype=colDefs)
        isocData['mass_init'] = massInit
        isocData['logL'] = logL
        isocData['logT'] = logT
        isocData['logg'] = logg
        isocData['ffco'] = ffco
        isocData['phase'] = phase
        isocData['wght'] = wght
        for (bandIdx,bandName,comment) in FILTER_LIST:
            isocData[bandName] = isocMags[:,bandIdx-1]
        print "isocMags.shape", isocMags.shape
        return isocData



if __name__ == '__main__':
    main()


