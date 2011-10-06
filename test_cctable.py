import numpy as np

import cctable
import fsps

def main():
    #library = MonteCarloLibrary("test_cctable")
    #library.reset()
    #library.define_samples(n=1000)
    #library.compute_models(nThreads=6, maxN=100)
    #library.create_table("test_cctable.h5", clobber=True)
    ccTable = cctable.CCTable("test_cctable.h5")
    #ccTable.make("megacam_gi_iK", ("MegaCam_g","MegaCam_i"),
    #        ("MegaCam_i","TMASS_Ks"), binsize=0.05, clobber=True)
    #ccTable.mass_light_table()
    ccTable.open("megacam_gi_iK")


class MonteCarloLibrary(fsps.FSPSLibrary):
    """docstring for MonteCarloLibrary"""
    
    def define_samples(self, n=50000):
        """Define the set of models."""
        for i in xrange(n):
            pset = fsps.ParameterSet(None, # automatically create a name
                sfh=4, # delayed tau
                tage=13.7, # select only modern-day observations
                imf_type=1, # Chabrier 2003
                dust_type=2, # Calzetti 2000 attenuation curve
                zmet=int(self._sample_zmet()),
                tau=float(self._sample_tau()),
                const=float(self._sample_const()),
                sf_start=float(self._sample_sf_start()),
                fburst=float(self._sample_fburst()),
                tburst=float(self._sample_tburst()),
                dust2=float(self._sample_dust2()),
                )
            self.register_pset(pset)

    def _sample_zmet(self):
        """Returns a random metallicity"""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Returns a random e-folding of SFR"""
        return np.random.uniform(0.1, 100.)
    
    def _sample_const(self):
        """Returns fraction of mass formed as a constant mode of SF"""
        return np.random.uniform(0.1,1.)
    
    def _sample_sf_start(self):
        """Start time of SFH in Gyr"""
        return np.random.uniform(0.5,13.)
    
    def _sample_fburst(self):
        """Fraction of mass formed in an instantaneous burst of SF."""
        return np.random.uniform(0.,1.)
    
    def _sample_tburst(self):
        """Time of the burst after the BB."""
        return np.random.uniform(1.5,13.5)
    
    def _sample_dust2(self):
        """Optical depth of ambient ISM."""
        return np.random.uniform(0.1,3.)

if __name__ == '__main__':
    main()
