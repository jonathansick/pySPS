"""Test fspsq with a tiny grid of models--a basic burn in test case."""

import fspsq
from fspsq import FSPSLibrary
from fspsq import ParameterSet

def main():
    tinyLibrary = TinySSPGrid("tinyssp", dbname="fsps")
    tinyLibrary.reset()
    tinyLibrary.generate_grid()
    tinyLibrary.compute_models(nThreads=1)

class TinySSPGrid(FSPSLibrary):
    """A small grid SSPs for three metallicities."""
    
    def generate_grid(self):
        """Create the model grid."""
        zmets = [1,2,3]
        for i, zmet in enumerate(zmets):
            modelName = "model%i" % i
            pset = ParameterSet(modelName, sfh=0, zmet=zmet)
            self.register_pset(pset)
            
if __name__ == '__main__':
    main()