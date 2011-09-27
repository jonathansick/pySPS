"""Set up a test grid of single-metallicity CSPs with different taus, and
difference metallicities.
"""
import numpy as np
import fspsq
from fspsq import FSPSLibrary
from fspsq import ParameterSet

def main():
    mediumLibrary = MediumGrid("mediumgrid", dbname="fsps")
    mediumLibrary.reset()
    mediumLibrary.generate_grid()
    mediumLibrary.compute_models(nThreads=6, maxN=50)

class MediumGrid(FSPSLibrary):
    """A small grid SSPs for three metallicities."""
    
    def generate_grid(self):
        """Create the model grid."""
        zmets = range(1,23)
        taus = np.arange(0.1, 10, 0.1)
        print zmets
        print taus
        i = 0
        for zmet in zmets:
            for tau in taus:
                modelName = "model%i" % i
                pset = ParameterSet(modelName, sfh=1, zmet=zmet, tau=float(tau))
                self.register_pset(pset)
                i += 1
        print "There are %i models" % self.count_models()
            
if __name__ == '__main__':
    main()