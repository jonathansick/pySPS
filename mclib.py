"""Basic Monte Carlo stellar population library."""

import numpy as np
import bson

import fspsq
from fspsq import FSPSLibrary
from fspsq import ParameterSet

def main():
    library = MonteCarloLibrary("mc.1", dbname='fsps')
    # library.reset()
    # library.define_samples()
    library.compute_models(nThreads=8, maxN=50)

class MonteCarloLibrary(FSPSLibrary):
    """Demonstration of a Monte Carlo sampled stellar pop library."""
    
    def define_samples(self, n=50000):
        """Define the set of models."""
        for i in xrange(n):
            pset = ParameterSet(None, # automatically create a name
                sfh=1,
                imf_type=1, # Chabrier 2003
                dust_type=2., # Calzetti 2000 attenuation curve
                zmet=int(self._sample_zmet()),
                tau=float(self._sample_tau()),
                const=float(self._sample_const()),
                sf_start=float(self._sample_sf_start()),
                fburst=float(self._sample_fburst()),
                tburst=float(self._sample_tburst()),
                dust2=float(self._sample_dust2())
                )
            self.register_pset(pset)
    
    def _sample_zmet(self):
        """Returns a random metallicity"""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Returns a random e-folding of SFR"""
        return np.random.uniform(0.1, 10.)
    
    def _sample_const(self):
        """Returns fraction of mass formed as a constant mode of SF"""
        return np.random.uniform(0.0,0.1)
    
    def _sample_sf_start(self):
        """Start time of SFH in Gyr"""
        return np.random.uniform(0.,7.)
    
    def _sample_fburst(self):
        """Fraction of mass formed in an instantaneous burst of SF."""
        return np.random.uniform(0.,2.)
    
    def _sample_tburst(self):
        """Time of the burst after the BB."""
        return np.random.uniform(0.,12.5)
    
    def _sample_dust2(self):
        """Optical depth of ambient ISM."""
        return np.random.uniform(0.1,1.)

if __name__ == '__main__':
    main()