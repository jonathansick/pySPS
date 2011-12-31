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
        print "Total mass*weight*dM", np.sum(isocData['mass_init']*isocData['wght']*dM)


if __name__ == '__main__':
    main()


