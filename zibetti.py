#!/usr/bin/env python
# encoding: utf-8
"""
This module attempts to replicate the Zibetti (2009) recipe for creating
a colour-colour M/L look up table. Of course, rather than use Charlot
& Bruzual's SPS code, the FSPS engine of Conroy, Gunn and White (2009) is used.

History
-------
2011-10-09 - Created by Jonathan Sick

"""
import numpy as np

from fsps import FSPSLibrary
from fsps import ParameterSet

TUNIVERSE = 13.7 # age of universe supposed in Gyr

def main():
    define_library = True
    compute_models = True
    make_table = True

    library = ZibettiLibrary("zibetti")
    if define_library:
        library.reset()
        library.define_samples()
    if compute_models:
        library.compute_models(nThreads=8, maxN=100)
    if make_table:
        library.create_table("zibetti.h5", clobber=True)

class ZibettiLibrary(FSPSLibrary):
    """A Monte Carlo stellar population library designed around the Zibetti
    (2009) priors for SFH and dust. The exact forms of the probability
    distribution functions are best specified in da Cunha, Charlot and Elbaz
    (2008).
    """
    def define_samples(self, n=50000):
        """Define the set of models."""
        for i in xrange(n):
            pset = ParameterSet(None, # automatically create a name
                sfh=1, # tau SFH
                tage=TUNIVERSE, # select only modern-day observations
                imf_type=1, # Chabrier 2003
                dust_type=2, # Calzetti 2000 attenuation curve
                zmet=int(self._sample_zmet()),
                tau=float(self._sample_tau()),
                const=float(0.), # no constant SF component
                sf_start=float(self._sample_sf_start()),
                fburst=float(self._sample_fburst()),
                tburst=float(self._sample_tburst()),
                dust1=float(self._sample_dust1()),
                dust2=float(self._sample_dust2()),
                )
            self.register_pset(pset)
    
    def _sample_zmet(self):
        """Returns a random metallicity from the Padova catalog."""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Power law timescale of SFH, :math:`SFR(t) \propto exp(-t/\tau)`.

        Note that users of the Charlot and Bruzual use a :math:`\gamma`
        parameter, where :math:`\gamma \equiv 1/\tau`.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\gamma) = 1-\tanh (8\gamma - 6)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(8.*x - 6):
            x = np.random.uniform(0.,1.)
            u = np.random.uniform(0.,2.)
        return x

    def _sample_sf_start(self):
        """Start of star-formation (Gyr. Defined in Kauffmann 2003."""
        return np.random.uniform(0.1, TUNIVERSE-1.5)

    def _sample_tburst(self, tform):
        """Time when the star-burst happens. We take the start of
        star-formation as an input so that there *can* always be a burst.

        This mathematical form of this prior used by Kauffmann et al is
        poorly specified. I quote:

        .. Bursts occur with equal probability at all times after tform and
           we have set the probability so that 50 per cent of the galaxies in
           the library have experienced a burst over the past 2 Gyr.

        The first part of that sentence does not necessarily imply the other.
        Regardless, I simply use a prior that bursts can happend uniformly
        between the start of star foramtion and the modern day.
        """
        return np.random.uniform(tform, TUNIVERSE)

    def _sample_fburst(self):
        """The fraction of stellar mass formed in a burst mode. Kauffmann
        logarithmically sample between 0 and 0.75.
        """
        return np.random.uniform(0., 0.75) # we're cheap and use uniform

    def _sample_dust1(self):
        """Sample the attenuation of young stellar light.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\tau_V) = 1-\tanh (1.5\tau_V - 6.7)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(1.5*x - 6.7):
            x = np.random.uniform(0.,6.)
            u = np.random.uniform(0.,2.)
        return x

    def _sample_dust2(self):
        """Sample the attenuation due to the ambient ISM.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\mu) = 1-\tanh (8 \mu - 6)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(8.*x - 6.):
            x = np.random.uniform(0.,6.)
            u = np.random.uniform(0.,2.)
        return x
    
if __name__ == '__main__':
    main()


